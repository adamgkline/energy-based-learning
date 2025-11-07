import sys
import torch

class CLTrainer(Epoch):
    
    def __init__(self, network, cost_fn, params, dataloader, differentiator, optimizer, energy_minimizer):
        """Initializes an instance of Trainer

        Args:
            network (Network): the network to train
            cost_fn (CostFunction): the cost function to optimize
            dataloader (Dataloader): the dataset on which to train the network
            differentiator (GradientEstimator): either EquilibriumProp or Backprop
            optimizer (str): the optimizer used to optimize.
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference
        """


        Epoch.__init__(self, 2)

        self._network = network
        self._params = params + cost_fn.params()  # FIXME
        self._cost_fn = cost_fn
        self._dataloader = dataloader
        self._differentiator = differentiator
        self._optimizer = optimizer
        self._energy_minimizer = energy_minimizer
    
    def run(self, verbose=False):
        """Train the model for one epoch over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: False.
        """

        self._reset()  # sets all the statistics to zero

        for x, y in self._dataloader:
            
            # grads = self._differentiator.compute(gradient)
            
            # # Free phase
            # self._network.unclamp()
            # self._network.set_inputs([x], reset=False)
            # self._free_energy_minimizer.compute_equilibrium()
            # self._cost_fn.set_target(y)
            # self._do_measurements(0)

            # # Clamped phase
            # self._network.clamp()
            # self._network.set_inputs([x, y], reset=False)
            # grads = self._differentiator.compute_gradient()
            # for

            # # inference (free phase relaxation)
            # self._network.set_input(x, reset=False)  # we set the input, and we let the state of the network where it was at the end of the previous batch
            # self._energy_minimizer.compute_equilibrium()  # we let the network settle to equilibrium (free state)
            # self._cost_fn.set_target(y)  # we present the correct (desired) output
            # self._do_measurements(0)  # we measure the statistics of the free state (energy value, cost value, error value, ...)

            # training step
            grads = self._differentiator.compute_gradient()  # compute the parameter gradients
            for param, grad in zip(self._params, grads): param.state.grad = grad  # Set the gradients of the parameters
            self._do_measurements(1)  # measure the statistics of training
            self._optimizer.step()  # perform one step of gradient descent on the parameters (of both the energy function E and the cost function C)
            for param in self._params: param.clamp_()  # clamp the parameters' states in their range of permissible values, if adequate

            if verbose:  # log the characteristics of training for the current epoch, up to the current mini-batch
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(str(self))
            sys.stdout.write('\n')

    def __str__(self):
        return 'TRAIN -- ' + Epoch.__str__(self)