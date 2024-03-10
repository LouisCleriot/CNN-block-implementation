# Implementation of layers of CNN in PyTorch

## Download toy dataset

Use the script get_datasets.sh in the datasets folder to download the toy dataset.

```bash
datasets/get_datasets.sh
```

## Layers implemented

All the layers are implemented in CNNBlocks.py. The layers implemented are:

- ResidualBlock
- DenseBlock
- BottleneckBlock
- SqueezeExcitationBlock
- InceptionBlock

## models

The models are implemented in models.py. The models implemented are:

- ResNet
- DenseNet
- InceptionNet
- Custom model called CustomNet

## Training

The training script is train_net.py. The script is originally written by Mamadou Mountagha BAH & Pierre-Marc Jodoin of the University of Sherbrooke and used in IFT780 "Neural Network" for practicals. The script is modified to train more diverse models.