import argparse
import numpy as np
import torch

from energy_based_learning.datasets import load_dataloaders
from energy_based_learning.model.resistive.network import DeepResistiveEnergy
from energy_based_learning.model.forward.network import ReLUNeuralNet
from energy_based_learning.model.function.network import Network
from energy_based_learning.model.function.cost import SquaredError
from energy_based_learning.model.resistive.minimizer import QuadraticMinimizer
from energy_based_learning.model.forward.minimizer import ForwardPass
from energy_based_learning.training.sgd import AugmentedFunction, EquilibriumProp, CoupledLearning
from energy_based_learning.training.sgd import Backprop
from energy_based_learning.training.epoch import Trainer, Evaluator
from energy_based_learning.training.monitor import Monitor, Optimizer
project_folder = '~/Documents/Projects/energy-based-learning'

# Handle command line arguments
parser = argparse.ArgumentParser(description='Simple implementation of coupled learning in a DRN')
parser.add_argument('--dataset', type = str, default = 'MNIST', help="The dataset used. Either `MNIST', `KMNIST' or `FMNIST'")
parser.add_argument('--model', type = str, default = 'DRN-CL', help="The network model. Either 'DRN-EP' (Deep Resistive Network trained with Equilibrium Propagation), 'DRN-CL' (Deep Resistive Network trained with Coupled learning)")
parser.add_argument('--layers', type = int, default = 1, help="The number of layers. Either 1, 2 or 3.")

args = parser.parse_args()

if __name__ == "__main__":
    
    dataset = args.dataset
    model = args.model
    num_layers = args.layers

    if dataset == 'MNIST':
            dataset_gain = 8.
    elif dataset == 'FMNIST':
        dataset = 'FashionMNIST'
        dataset_gain = 5.
    elif dataset == 'KMNIST':
        dataset = 'KuzushijiMNIST'
        dataset_gain = 5.
    else:
        raise ValueError("expected 'MNIST', 'KMNIST' or 'FMNIST' but got {}".format(model))

    # Hyperparameters
    if model[:3] == 'DRN':
        if num_layers == 1:
            layer_shapes = [(2, 28, 28), (1024,), (10,)]
            model_gain = 60.
            num_iterations_inference = 4
            num_iterations_training = 4
            learning_rates_weights = [0.005, 0.005]
            learning_rates_biases = [0.005, 0.005]
            nudging = 0.5  # for EP only
        elif num_layers == 2:
            layer_shapes = [(2, 28, 28), (1024,), (1024,), (10,)]
            model_gain = 250.
            num_iterations_inference = 5
            num_iterations_training = 5
            learning_rates_weights = [0.002, 0.006, 0.005]
            learning_rates_biases = [0.002, 0.006, 0.005]
            nudging = 1.0  # for EP only
        elif num_layers == 3:
            layer_shapes = [(2, 28, 28), (1024,), (1024,), (1024,), (10,)]
            model_gain = 500.
            num_iterations_inference = 6
            num_iterations_training = 6
            learning_rates_weights = [0.005, 0.02, 0.08, 0.005]
            learning_rates_biases = [0.0005, 0.002, 0.008, 0.0005]
            nudging = 2.0  # for EP only
        else:
            raise ValueError("expected 1, 2 or 3 but got {}".format(num_layers))
    else:
        raise ValueError("expected 'DRN-EP' or 'DRN-CL' but got {}".format(model))
    
    # Load the training and test data
    # batch_size = 32
    batch_size = 2
    training_loader, test_loader = load_dataloaders(dataset, batch_size, augment_32x32=False, normalize=False)

    # Build the energy function (DRN or ReLU NN)
    weight_gains = [1.0] * (len(layer_shapes) - 1)
    if model[:3] == 'DRN':
        input_gain = dataset_gain * model_gain
        energy_fn = DeepResistiveEnergy(layer_shapes, weight_gains, input_gain)
    else:
        raise ValueError("expected 'DRN-EP' or 'DRN-CL' but got {}".format(model))

    # Set the device on which we run and train the network
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"   # "mps" for mac, typically "cuda" elsewhere
    energy_fn.set_device(device)

    # Define the cost function (mean squared error)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)

    # Define the network
    network = Network(energy_fn)
    free_layers = network.free_layers()
    params = energy_fn.params()
    layers = energy_fn.layers()

    # Build the energy minimizers and gradient estimator
    if model == 'DRN-EP':
        # Define the energy minimizer at inference
        energy_minimizer_inference = QuadraticMinimizer(energy_fn, free_layers)
        energy_minimizer_inference.num_iterations = num_iterations_inference
        energy_minimizer_inference.mode = 'forward'

        # Define the energy minimizer during training
        augmented_fn = AugmentedFunction(energy_fn, cost_fn) # This adds
        energy_minimizer_training = QuadraticMinimizer(augmented_fn, free_layers)
        energy_minimizer_training.num_iterations = num_iterations_training
        energy_minimizer_training.mode = 'backward'

        # Define the gradient estimator (equilibrium propagation)
        estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
        estimator.nudging = nudging
        estimator.variant = 'centered'

    elif model == 'DRN-CL':
        # Define the energy minimizer at inference
        energy_minimizer_inference = QuadraticMinimizer(energy_fn, free_layers)
        energy_minimizer_inference.num_iterations = num_iterations_inference
        energy_minimizer_inference.mode = 'forward' 

        # Define the energy minimizer during training
        augmented_fn = AugmentedFunction(energy_fn, cost_fn)
        energy_minimizer_training = QuadraticMinimizer(augmented_fn, free_layers)
        energy_minimizer_training.num_iterations = num_iterations_training
        energy_minimizer_training.mode = 'backward'

        # Define the gradient estimator (equilibrium propagation)
        estimator = CoupledLearning(params, layers, energy_fn, energy_minimizer_training, nudging)
        estimator.nudging = nudging

    # Build the optimizer (SGD)
    learning_rates = learning_rates_biases + learning_rates_weights
    momentum = 0.9
    weight_decay = 0.
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    trainer = Trainer(network, cost_fn, params, training_loader, estimator, optimizer, energy_minimizer_inference)
    evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)
    
    # Define the scheduler for the learning rates
    num_epochs = 100
    # gamma = 0.99  # value used in original
    gamma = 1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

     # Define the path and the monitor to perform the run
    path = '/'.join(['tests/EP-vs-CL', model, str(num_layers)])
    monitor = Monitor(energy_fn, cost_fn, trainer, scheduler, evaluator, path)

    # Print the characteristics of the run
    print('Dataset: {} -- batch_size={}'.format(dataset, batch_size))
    print('Network: ', energy_fn)
    print('Cost function: ', cost_fn)
    print('Energy minimizer during inference: ', energy_minimizer_inference)
    print('Energy minimizer during training: ', energy_minimizer_training)
    print('Gradient estimator: ', estimator)
    print('Parameter optimizer: ', optimizer)
    print('Number of epochs = {}'.format(num_epochs))
    print('Path = {}'.format(path))
    print('Device = {}'.format(device))
    print()

    # Launch the experiment
    monitor.run(num_epochs, verbose=True)

    for param in params:
        (std, mean) = torch.std_mean(param.state.to('cpu'))
        maxi = torch.max(param.state.to('cpu'))
        print('{}: mean={:.5f} std={:.5f}, max={:.5f}'.format(param.name, mean, std, maxi))