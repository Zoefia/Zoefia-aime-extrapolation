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
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

## Structure
All the datasets should be placed under `datasets/` and the pretrained models should be placed under `pretrained-models/`. 
Results will be placed under `logs/`, which you can view by tensorboard.

## Download the datasets and pre-trained models
The datasets and pretrained models can be access from [Github Release](https://github.com/argmax-ai/aime/releases/latest). All the datasets and models are released under a [_CC BY 4.0 license_](https://creativecommons.org/licenses/by/4.0/). For more details, please check out the [Data Card](datasets/readme.md) and [Model Card](pretrained-models/readme.md).

For datasets, you need to extract it to `datasets/` folder by
```
tar -xzvf <.tgz file> -C datasets/
```

To generate the `walker-mix` dataset, please download all the walker datasets and then run the following command:

```
python scripts/mix_datasets.py -i walker-random walker-plan2explore-buffer walker-stand-buffer walker-walk-buffer -o walker-mix
```

For pretrain models, you need to extract it to `pretrained-models/` folder by
```
tar -xzvf pretrained-models.tgz -C pretrained-models/
`