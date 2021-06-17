
from typing import Dict, Optional

import numpy as np
import torch
from einops import rearrange

from aime.data import ArrayDict
from aime.dist import Normal, TanhNormal, kl_divergence

from .base import MIN_STD, MLP, decoder_classes, encoder_classes
from .policy import TanhGaussianPolicy


class SSM(torch.nn.Module):
    """
    State-Space Model BaseClass.
    NOTE: in some literature, this type of model is also called sequential auto-encoder or stochastic rnn.
    NOTE: in the current version, SSM also contain encoders, decoders and probes for the
          sack of simplicity for model training. But this may not be the optimal modularization.
    """  # noqa: E501

    def __init__(
        self,
        input_config,
        output_config,
        action_dim,
        state_dim,
        probe_config=None,
        intrinsic_reward_config=None,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        kl_scale=1.0,
        free_nats=0.0,
        kl_rebalance=None,
        nll_reweight="dim_wise",
        idm_mode="none",
        min_std=None,
        **kwargs,
    ) -> None:
        """
        input_config            : a list of tuple(name, dim, encoder_config)
        output_config           : a list of tuple(name, dim, decoder_config)
        action_dim              : int
        state_dim               : config for dims of the latent state
        probe_config            : a list of tuple(name, dim, decoder_config)
        intrinsic_reward_config : an optional config to enable the intrinsic reward based on plan2explore paper
        hidden_size             : width of the neural networks
        hidden_layers           : depth of the neural networks
        norm                    : what type of normalization layer used in the network, default is `None`.
        kl_scale                : scale for the kl term in the loss function
        free_nats               : free information per dim in latent space that is not penalized
        kl_rebalance            : rebalance the kl term with the linear combination of the two detached version, default is `None` meaning disable, enable with a float value between 0 and 1
        nll_reweight            : reweight method for the likelihood term (also the kl accordingly), choose from `none`, `modility_wise`, `dim_wise`.
        idm_mode                : mode for idm, choose from `none`, `end2end` and `detach`
        min_std                 : the minimal std for all the learnable distributions for numerical stablity, set to None will follow the global default.

        NOTE: For output and probe configs, there can be a special name `emb` which indicate to predict the detached embedding from the encoders.
              For that use case, the `dim` in that config tuple will be overwrite.
        """  # noqa: E501
        super().__init__()
        self.input_config = input_config
        self.output_config = output_config
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.probe_config = probe_config
        self.intrinsic_reward_config = intrinsic_reward_config
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.norm = norm
        self.kl_scale = kl_scale
        self.free_nats = free_nats
        self.kl_rebalance = kl_rebalance
        self.nll_reweight = nll_reweight
        assert self.nll_reweight in ("none", "modility_wise", "dim_wise")
        self.idm_mode = idm_mode
        assert self.idm_mode in (
            "none",
            "end2end",
            "detach",
        ), f"recieved unknown idm_mode `{self.idm_mode}`."
        self.min_std = min_std if min_std is not None else MIN_STD

        self.create_network()
        self.metric_func = torch.nn.MSELoss()

    def create_network(
        self,
    ):
        self.encoders = torch.nn.ModuleDict()
        for name, dim, encoder_config in self.input_config:
            encoder_config = encoder_config.copy()
            encoder_type = encoder_config.pop("name")
            self.encoders[name] = encoder_classes[encoder_type](dim, **encoder_config)

        self.emb_dim = sum(
            [encoder.output_dim for name, encoder in self.encoders.items()]
        )

        self.decoders = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.output_config:
            if name == "emb":
                dim = self.emb_dim
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            self.decoders[name] = decoder_classes[decoder_type](
                self.state_feature_dim, dim, **decoder_config
            )

        self.probes = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.probe_config:
            if name == "emb":
                dim = self.emb_dim
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            self.probes[name] = decoder_classes[decoder_type](
                self.state_feature_dim, dim, **decoder_config
            )

        if not (self.idm_mode == "none"):
            # Inverse Dynamic Model (IDM) is a non-casual policy
            self.idm = TanhGaussianPolicy(
                self.state_feature_dim + self.emb_dim,
                self.action_dim,
                self.hidden_size,
                self.hidden_layers,
            )

        if self.intrinsic_reward_config is not None:
            self.emb_prediction_heads = torch.nn.ModuleList(
                [
                    MLP(
                        self.state_feature_dim + self.action_dim,
                        self.emb_dim,
                        self.intrinsic_reward_config["hidden_size"],
                        self.intrinsic_reward_config["hidden_layers"],
                    )
                    for _ in range(self.intrinsic_reward_config["num_ensembles"])
                ]
            )

    def reset(self, batch_size):
        """reset the hidden state of the SSM"""
        raise NotImplementedError

    @property
    def state_feature_dim(self):
        return self.state_dim

    def stack_states(self, states):
        return ArrayDict.stack(states, dim=0)

    def flatten_states(self, states):
        # flatten the sequence of states as the starting state of rollout
        if isinstance(states, list):
            states = self.stack_states(states)
        states.vmap_(lambda v: rearrange(v, "t b f -> (t b) f"))
        return states

    def get_state_feature(self, state):
        return state

    def get_emb(self, obs):
        return torch.cat(
            [model(obs[name]) for name, model in self.encoders.items()], dim=-1
        )

    def get_output_dists(self, state_feature, names=None):
        if names is None:
            names = self.decoders.keys()
        return {name: self.decoders[name](state_feature) for name in names}

    def get_outputs(self, state_feature, names=None):
        if names is None:
            names = self.decoders.keys()
        return {name: self.decoders[name](state_feature).mode for name in names}

    def get_probe_dists(self, state_feature, names=None):
        if names is None:
            names = self.probes.keys()
        return {name: self.probes[name](state_feature) for name in names}

    def get_probes(self, state_feature, names=None):
        if names is None:
            names = self.probes.keys()
        return {name: self.probes[name](state_feature).mode for name in names}

    def compute_kl(self, posterior, prior):
        if self.kl_rebalance is None:
            return kl_divergence(posterior, prior)
        else:
            return self.kl_rebalance * kl_divergence(posterior.detach(), prior) + (
                1 - self.kl_rebalance