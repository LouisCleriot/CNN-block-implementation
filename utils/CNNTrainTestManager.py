# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
Modified: Louis CLERIOT
License:
Other: Suggestions are welcome
"""

import warnings
import torch
import numpy as np
from utils.DataManager import DataManager
from typing import Callable, Type
from tqdm import tqdm
import matplotlib.pyplot as plt


class CNNTrainTestManager(object):
    """
    Class used the train and test the given model in the parameters 
    """

    def __init__(self, model,
                 trainset: torch.utils.data.Dataset,
                 testset: torch.utils.data.Dataset,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 batch_size=1,
                 validation=None,
                 use_cuda=False,
                 metric="accuracy"):
        """
        Args:
            model: model to train
            trainset: dataset used to train the model
            testset: dataset used to test the model
            loss_fn: the loss function used
            optimizer_factory: A callable to create the optimizer. see optimizer function
            below for more details
            validation: wether to use custom validation data or let the one by default
            use_cuda: to Use the gpu to train the model
        """

        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'
        self.device = torch.device(device_name)
        if validation is not None:
            self.data_manager = DataManager(trainset, testset, batch_size=batch_size, validation=validation)
        else:
            self.data_manager = DataManager(trainset, testset, batch_size=batch_size)
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer_factory(self.model)
        self.model = self.model.to(self.device)
        self.use_cuda = use_cuda
        if metric == "accuracy":
            self.metric = self.accuracy
        elif metric == "f1":
            self.metric = self.f1_score
        self.metric_values = {}

    def train(self, num_epochs):
        """
        Train the model for num_epochs times
        Args:
            num_epochs: number times to train the model
        """
        # Initialize metrics container
        self.metric_values['train_loss'] = []
        self.metric_values['train_acc'] = []
        self.metric_values['val_loss'] = []
        self.metric_values['val_acc'] = []

        # Create pytorch's train data_loader
        train_loader = self.data_manager.get_train_set()
        # train num_epochs times
        for epoch in range(num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0

            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                train_accuracies = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, train_labels = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)
                    # computes loss using loss function loss_fn
                    loss = self.loss_fn(train_outputs, train_labels)

                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    """
                    Way it could be done manually

                    with torch.no_grad():
                        for param in self.model.parameters():
                            param -= learning_rate * param.grad
                    """
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())
                    train_accuracies.append(self.metric(train_outputs, train_labels))

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()
            # evaluate the model on validation data after each epoch
            self.metric_values['train_loss'].append(np.mean(train_losses))
            self.metric_values['train_acc'].append(np.mean(train_accuracies))
            self.evaluate_on_validation_set()

        print("Finished training.")

    def evaluate_on_validation_set(self):
        """
        function that evaluate the model on the validation set every epoch
        """
        # switch to eval mode so that layers like batchnorm's layers nor dropout's layers
        # works in eval mode instead of training mode
        self.model.eval()

        # Get validation data
        val_loader = self.data_manager.get_validation_set()
        validation_loss = 0.0
        validation_losses = []
        validation_accuracies = []

        with torch.no_grad():
            for j, val_data in enumerate(val_loader, 0):
                # transfer tensors to the selected device
                val_inputs, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)

                # forward pass
                val_outputs = self.model(val_inputs)

                # compute loss function
                loss = self.loss_fn(val_outputs, val_labels)
                validation_losses.append(loss.item())
                validation_accuracies.append(self.metric(val_outputs, val_labels))
                validation_loss += loss.item()

        self.metric_values['val_loss'].append(np.mean(validation_losses))
        self.metric_values['val_acc'].append(np.mean(validation_accuracies))

        # displays metrics
        print('Validation loss %.3f' % (validation_loss / len(val_loader)))

        # switch back to train mode
        self.model.train()

    def accuracy(self, outputs, labels):
        """
        Computes the accuracy of the model
        Args:
            outputs: outputs predicted by the model
            labels: real outputs of the data
        Returns:
            Accuracy of the model
        """
        predicted = outputs.argmax(dim=1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)
    
    def f1_score(self, outputs, labels):
        """
        Computes the f1_score of the model
        Args:
            outputs: outputs predicted by the model
            labels: real outputs of the data
        Returns:
            f1_score of the model
        """
        predicted = outputs.argmax(dim=1)
        tp = (predicted * labels).sum().item()
        fp = ((predicted - labels) == 1).sum().item()
        fn = ((predicted - labels) == -1).sum().item()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

    def evaluate_on_test_set(self):
        """
        Evaluate the model on the test set
        :returns;
            Accuracy of the model on the test set
        """
        test_loader = self.data_manager.get_test_set()
        accuracies = 0
        with torch.no_grad():
            for data in test_loader:
                test_inputs, test_labels = data[0].to(self.device), data[1].to(self.device)
                test_outputs = self.model(test_inputs)
                accuracies += self.metric(test_outputs, test_labels)
        print("Accuracy on the test set: {:05.3f} %".format(100 * accuracies / len(test_loader)))

    def plot_metrics(self, out_file_name='fig.png'):
        """
        Function that plots train and validation losses and accuracies after training phase
        """
        epochs = range(1, len(self.metric_values['train_loss']) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(epochs, self.metric_values['train_loss'], '-o', label='Training loss')
        ax1.plot(epochs, self.metric_values['val_loss'], '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(epochs, self.metric_values['train_acc'], '-o', label='Training accuracy')
        ax2.plot(epochs, self.metric_values['val_acc'], '-o', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        f.savefig(out_file_name)
        plt.show()



def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f
