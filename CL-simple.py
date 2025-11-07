import argparse
import numpy as np
import torch
import torch.nn as nn

from energy_based_learning.datasets import load_dataloaders
from energy_based_learning.model.resistive.network import DeepResistiveEnergy
from energy_based_learning.model.function.network import Network
from energy_based_learning.model.resistive.minimizer import QuadraticMinimizer
from energy_based_learning.model.forward.minimizer import ForwardPass
from energy_based_learning.training.sgd import CoupledLearning
from energy_based_learning.training.epoch import Trainer, Evaluator
from energy_based_learning.training.monitor import Monitor, Optimizer
project_folder = '~/Documents/Projects/energy-based-learning'




if __name__ == "__main__":
    
    dataset = 'MNIST'
    dataset_gain = 8.

    num_layers = 1
    layer_shapes = [(2, 28, 28), (1024,), (10,)]
    model_gain = 60.
    num_iterations_inference = 4
    num_iterations_training = 4
    learning_rates_weights = [0.005, 0.005]
    learning_rates_biases = [0.005, 0.005]
    nudging = 0.5 

    # Load the training and test data
    batch_size = 2
    training_loader, test_loader = load_dataloaders(dataset, batch_size, augment_32x32=False, normalize=False)

    # Build the energy function
    weight_gains = [1.0] * (len(layer_shapes) - 1)
    input_gain = dataset_gain * model_gain
    energy_fn = DeepResistiveEnergy(layer_shapes, weight_gains, input_gain)

    # Set the device on which we run and train the network
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"   # "mps" for mac, typically "cuda" elsewhere
    energy_fn.set_device(device)



    size = len(training_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(training_loader):
        X, y = X.to(device), y.to(device)

        # free phase
        model.input_layers = [input_layer]
        model.clamp([X])
        y_free = model.read(output_layer)
        power_free = model.power()

        # clamped phase
        model.input_layers = [input_layer, output_layer]
        y_nudged = y_free * nudging + (1 - nudging) * y
        model.clamp([X, y_nudged])
        power_clamped = model.power()

        loss = (power_free - power_clamped) / model.nudging

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    # set coupled learning loss
    loss = power_free - power_clamped       # Tensor with shape ...

    
    


class ResistorDiodeNetwork(nn.Module):
    
    def __init__(self, layer_shapes, weight_gains, input_gain):
        super().__init__()

        self.input_gain = input_gain
        self.weight_gains = weight_gains
        
        # build the layers of the network
        input_shape = layer_shapes[0]
        hidden_shapes = layer_shapes[1:-1]
        output_shape = layer_shapes[-1]
        
        # Make this a list of layers
        self.layers = [input_layer] + hidden_layers + [output_layer]
        self.input_layers = [input_layer]

    def clamp(self, inputs):
        """
        clamp the input layers at the specified tensors.
        
        inputs should be a List of Tensors.
        """

        for x, layer in zip(inputs, self.layers):
            layer.state = x

        # Run scellier code to minimize
        # How do we represent node states?

    def set_input_layers(self, layers):
        self.input_layers = layers


    def power(self, inputs):
        # network_state = self(inputs)    # Tensor
        self.clamp(inputs)

        return sum([])


    # def quadratic_function_argmin(self, a, b):
    #     """
    #     Returns the argmin of a quadratic function f(x) = a x^2 + b x + c
    #     """
    #     return - b / (2. * a)

    def forward(self, x):
        pass