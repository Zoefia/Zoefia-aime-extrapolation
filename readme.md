![argmax.ai](pic/argmaxlogo.png)

*This repository is published and maintained by the Volkswagen Group Machine Learning Research Lab.*

*Learn more at https://argmax.ai.*

## AIME 

This repository contains the original implementation of [AIME](https://openreview.net/forum?id=WjlCQxpuxU) in PyTorch.

![AIME Framework](pic/aime.jpg)

If you find this code useful, please reference in your paper:
```BibTeX
@inproceedings{
zhang2023aime,
title={Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models},
author={Xingyuan Zhang and Philip Becker-Ehmck and Patrick van der Smagt and Maximilian Karl},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=WjlCQxpuxU}
}
```

## Setup
```
conda create -n aime python=3.9
conda activate aime
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio=