import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils.CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from torchvision import datasets
from importlib.util import spec_from_file_location, module_from_spec 
import os
import models
from sys import exit

base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
dataset_directory = './datasets_class'
model_directory = './models'

class Trainer():
    def __init__ (self, model="CnnVanilla", dataset="cifar10", optimizer="Adam", batch_size=64, num_epochs=10, validation=0.1, learning_rate=0.01, data_aug=False, loss="ce", metric="accuracy", out="fig.png"):
        self.model = model
        self.dataset = dataset
        #check if dataset is in torchvision.datasets.
        if hasattr(datasets, dataset):
            print(f"Dataset {dataset} found in torchvision.datasets")
            # Download the train and test set and apply transform on it
            train_set = getattr(datasets, dataset)(root='./datasets', train=True, download=True, transform=base_transform)
            test_set = getattr(datasets, dataset)(root='./datasets', train=False, download=True, transform=base_transform)
            nb_classes = len(train_set.classes)
        else:
            print(f"Dataset {dataset} not found in torchvision.datasets")
            print("checking for custom dataset")
            #check if dataset is in datasets_class
            dataset_file = dataset + '.py'
            dataset_path = os.path.join(dataset_directory, dataset_file)
            if os.path.exists(dataset_path):
                print(f"Dataset {dataset} found in datasets_class")
                module_name = f"datasets_class.{dataset}"
                spec = spec_from_file_location(module_name, dataset_path)
                dataset_module = module_from_spec(spec)
                spec.loader.exec_module(dataset_module)
                train_set = dataset_module.train_set
                test_set = dataset_module.test_set
                nb_classes = dataset_module.nb_classes
            else:
                print(f"Dataset {dataset} not found in datasets_class")
                print("Please provide a valid dataset")
                exit()
        #check if optimizer is in torch.optim.
        if hasattr(optim, optimizer):
            print(f"Optimizer {optimizer} found in torch.optim")
            optimizer_class = getattr(optim, optimizer)
            optimizer_factory = optimizer_setup(optimizer_class, lr=learning_rate)
        else:
            print("Optimizer not found in torch.optim")
            print("Please provide a valid optimizer")
            exit()
        
        
        model_file = model + '.py'
        model_path = os.path.join(model_directory, model_file)
        if os.path.exists(model_path):
            print(f"Model {model} found in models")
            module_name = f"models.{model}"
            spec = spec_from_file_location(module_name, model_path)
            model_module = module_from_spec(spec)
            spec.loader.exec_module(model_module)
            
            model_class = getattr(model_module, model)
            model = model_class(num_classes=nb_classes)
            
        else:
            print(f"Model {model} not found in models")
            print("Please provide a valid model")
            exit()
        self.model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        loss_fn=nn.CrossEntropyLoss(),
                                        optimizer_factory=optimizer_factory,
                                        batch_size=batch_size,
                                        validation=validation,
                                        use_cuda=True,
                                        metric=metric)
        self.num_epochs = num_epochs
        self.out = out
        
    def train(self):
        print("Training {} on {} for {} epochs".format(self.model.__class__.__name__, self.dataset, self.num_epochs))
        self.model_trainer.train(self.num_epochs)
        self.model_trainer.plot_metrics(self.out)
        print("Training completed!")
    
    def test(self):
        self.model_trainer.evaluate_on_test_set()
        