
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