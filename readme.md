# Implementation of layers of CNN in PyTorch

## Datasets

You can use both pytorch or custom datasets.

### Pytorch datasets

There is scripts in the `datasets` folder to download datasets like CIFAR 10 and FASHIONMNISST, you can also just use the code from the trainer with the name form torchvision.datasets. and it will download the dataset for you if not already downloaded.

### Custom datasets

Create the dataset directory in the `datasets` folder. It should have the same format as `torchvision.datasets.ImageFolder` where the images are in the subdirectories of the dataset directory.

The script `utils/train_test_split.sh` can be used to split the dataset into training and testing datasets. The script will create a `Train` and `Test` directory in the dataset directory. Each class will have a subdirectory in the `Train` and `Test` directory.

## Layers implemented

Different layer are implemented from scratch using their scientific papers and pytorch. They are grouped into different categories :

- Basic blocks (Residual / Dense / ConvBottleneck)
- Attention (SqueezeAndExciteBlock / EfficientChannelAttention)
- Channels reduction (FireBlock / SlimeConv)
- Shuffle Blocks (InterleavedModule / ShuffleModule / InterleavedGroupConvolutionModule)
- Multi-scale
  - Inception (InceptionModuleV1 / InceptionModuleV2Base / InceptionModuleV2Factorize / InceptionModuleV2Wide / InceptionModulev2Pooling)
  - Channel wise (HierarshicalSplitBlock)
  - Dimension wise (MultiGridConv / ASPPModule)

## Training

The training is done using the `Trainer` class. It's instantiated in the notebook `train.ipynb`. The classe is in early stage
and will be improved in the future.(custom loss, choice of transform, ...)

## How to use the repository

1. Find the dataset you want, either in the pytorch datasets or create a custom dataset.
2. Create a model python file in the `models` using same structure as the other models.
3. Modify the notebook `train.ipynb` to modify the training parameters, model and dataset.
4. Run the notebook to train the model.
5. Visualize the results in the png figures