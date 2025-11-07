from abc import ABC, abstractmethod
from itertools import accumulate
import torch

from energy_based_learning.model.function.interaction import Function, SumSeparableFunction
from energy_based_learning.model.minimizer.minimizer import ParamUpdater, GradientDescentUpdater


class CLNudging(Function):
    def __init__(self, function):
        """Initializes an instance of Coupled Learning Nudging

        Args:
            function (Function): the function to scale by a nudging factor
        """
        self._function = function
        self._nudging = 0.

        Function.__init__(self, function.layers(), function.params())

    @property
    def nudging(self):
        """Get and sets the nudging value"""
        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        self._nudging = nudging

    def eval(self):
        """Value of the nudging function. This is the function's value times the nudging value.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the value of an example in the current mini-batch
        """
        return self._nudging * self._function.eval()

    def a_coef_fn(self, layer):
        """Returns the function that computes the coefficient a for a given layer"""
        a_coef_fn = self._function.a_coef_fn(layer)
        return lambda: self._nudging * a_coef_fn()

    def b_coef_fn(self, layer):
        """Returns the function that computes the coefficient b for a given layer"""
        b_coef_fn = self._function.b_coef_fn(layer)
        return lambda: self._nudging * b_coef_fn()


class CLAugmentedFunction(SumSeparableFunction):

    def __init__(self, energy_fn, cost_fn):
        """Creates an instance of CLAugmentedFunction"""

        layers = energy_fn.layers()
        params = energy_fn.params()

        input_layer = layers[0]
        output_layer = layers[-1]

        nudging = CLNudging(cost_fn)
        # interactions = [energy_fn, nudging]
        # SumSeparableFunction.__init__(self, layers, params, interactions)

        self._nudging = nudging
        self._energy_fn = energy_fn

    def clamp(self):
        self._energy_fn.clamp()

    def eval(self):
        pass


class CoupledLearning(GradientEstimator):
    """
    Class for estimating the parameter gradients of a cost function in an energy-based system via coupled learning (CL)

    Equilibrium propagation (EP) estimates the gradient (wrt to theta) of C(s(theta)) under the constraint that s(theta) = argmin_s E(theta,s)

    The EP gradient estimator depends on two scalars: nudging_1 and nudging_2. It is given by the formula
    estimator(nudging_1, nudging_2) := [ dE(theta,s(nudging_2))/dtheta - dE(theta,s(nudging_1))/dtheta ] / [ nudging_2 - nudging_1 ]
    where s(nudging) = argmin_s [ E(theta,s) + nudging * C(s) ]

    The scalars nudging_1 and nudging_2 are determined by the attributes 'variant' and 'nudging' of the EquilibriumProp class, as follows:
    * if the variant is 'positive', then nudging_1 = 0 and nudging_2 = + nudging
    * if the variant is 'negative', then nudging_1 = - nudging and nudging_2 = 0
    * if the variant is 'centered', then nudging_1 = - nudging and nudging_2 = + nudging

    Attributes
    ----------
    _params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
    _layers (list of Layers): the layers that minimize the augmented energy function E(theta,s) + nudging * C(s)
    _energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function (in the perturbed phase of EP)
    variant (str): either 'positive' (positively-perturbed EP), 'negative' (negatively-perturbed EP) or 'centered' (centered EP)
    nudging (float): the nudging value used to estimate the parameter gradients via EP
    use_alternative_formula (bool): which equilibrium propagation formula is used to estimating the parameter gradients. Either the 'standard' formula (False) or the 'alternative' formula (True)

    Methods
    -------
    compute_gradient()
        Compute and return the parameter gradients via EP
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- EP gradients
    """

    def __init__(self, params, layers, network, free_energy_minimizer, clamped_energy_minimizer, nudging=0.25):
        """Creates an instance of equilibrium propagation

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
            layers (list of Layers): the Layers that minimize the augmented energy function
            energy_fn (Function): the energy function
            cost_fn (Function): the cost function
            energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function
            variant (str, optional): either 'positive' (positively-perturbed EP), 'negative' (negatively-perturbed EP) or 'centered' (centered EP). Default: 'centered'
            nudging (float, optional): the nudging value used to estimate the parameter gradients via EP. Default: 0.25
            use_alternative_formula (bool, optional): which equilibrium propagation formula is used to estimating the parameter gradients. Either the 'standard' formula (False) or the 'alternative' formula (True). Default: False
        """

        self._params = params
        self._layers = layers

        self._param_updaters = [ParamUpdater(param, energy_fn) for param in params]
        # self._param_updaters_cost = [ParamUpdater(param, cost_fn) for param in cost_fn.params()]
        self._layer_updaters = [GradientDescentUpdater(layer, energy_fn) for layer in layers]  # FIXME: one should be using the layer updaters of the energy minimizer

        self._network = network
        self._augmented_fn = energy_fn  # AugmentedFunction(energy_fn, cost_fn)
        self._energy_minimizer = energy_minimizer

        self._nudging = nudging
        self._variant = variant
        self._set_nudgings()

        self._use_alternative_formula = use_alternative_formula

    @property
    def nudging(self):
        """Get and sets the nudging value used for estimating the parameter gradients via equilibrium propagation"""
        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        if nudging > 0.:
            self._nudging = nudging
            self._set_nudgings()
        else: raise ValueError("expected a positive nudging value, but got {}".format(nudging))

    @property
    def variant(self):
        """Get and sets the training mode ('positive', 'negative' or 'centered')"""
        return self._variant

    @variant.setter
    def variant(self, variant):
        if variant in ['positive', 'negative', 'centered']:
            self._variant = variant
            self._set_nudgings()
        else: raise ValueError("expected 'positive', 'negative' or 'centered' but got {}".format(variant))

    def compute_gradient(self):
        """Estimates the parameter gradient via coupled learning
        """

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        # cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters
        
        # First phase: compute the first equilibrium state of the layers
        layers_first = self._free_energy_minimizer.compute_equilibrium()

        # Second phase: compute the second equilibrium state of the layers
        for layer, state in zip(self._layers, layers_free): layer.state = state  # hack: we start the second phase from the `free state' again
        # self._augmented_fn.nudging = self._second_nudging
        # self._augmented_fn.clamp()
        self._network.clamp()
        self._network.set_inputs()
        layers_second = self._clamped_energy_minimizer.compute_equilibrium()

        # Compute the parameter gradients using CL 
        param_grads = self._standard_param_grads(layers_first, layers_second)

        # return param_grads + cost_grads
        return param_grads
    
    def detailed_gradients(self, cumulative=True):
        """Compute and return the sequence of time-dependent layer- and parameter- EP gradients

        Calling this method leaves the state of the layers unchanged
        For the method to return the correct detailed gradients of EP, the layers must be at their 'free state' (equilibrium for nudging=0) when calling the method

        Args:
            cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: True.

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        # First phase: compute the layers' activations along the first trajectory
        layers_free = [layer.state for layer in self._layers]  # we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)
        self._augmented_fn.nudging = self._first_nudging
        trajectory_first = self._energy_minimizer.compute_trajectory()
        
        # Second phase: compute the layers' activations along the second trajectory
        for layer, state in zip(self._layers, layers_free): layer.state = state  # we start over from the `free state' again
        self._augmented_fn.nudging = self._second_nudging
        trajectory_second = self._energy_minimizer.compute_trajectory()

        # Compute the layer gradients
        trajectory_first = [dict(zip(trajectory_first, v)) for v in zip(*trajectory_first.values())]  # transform the dictionary of lists into a list of dictionaries
        trajectory_second = [dict(zip(trajectory_second, v)) for v in zip(*trajectory_second.values())]  # transform the dictionary of lists into a list of dictionaries
        layer_grads = [self._layer_grads(first, second) for first, second in zip(trajectory_first[:-1], trajectory_second[:-1])]
        layer_grads = list(map(list, zip(*layer_grads)))  # transpose the list of lists: transform the time-wise layer-wise gradients into layer-wise time-wise gradients

        # Compute the parameter gradients with either the standard EquilibriumProp formula, or the alternative EquilibriumProp formula
        if self._use_alternative_formula:
            param_grads = [self._alternative_param_grads(layers_free, first, second) for first, second in zip(trajectory_first[1:], trajectory_second[1:])]
        else:
            param_grads = [self._standard_param_grads(first, second) for first, second in zip(trajectory_first[1:], trajectory_second[1:])]
        param_grads = list(map(list, zip(*param_grads)))  # transpose the list of lists: transform the time-wise parameter-wise gradients into parameter-wise time-wise gradients

        # Store the layer-wise and parameter-wise time-wise gradients in a dictionary
        grads = dict()
        for layer, gradients in zip(self._layers, layer_grads): grads[layer.name] = gradients
        for param, gradients in zip(self._params, param_grads): grads[param.name] = gradients
        
        # Transform the time-wise gradients into time-wise increases if required
        if not cumulative:
            for layer in self._layers:
                layer_grads = grads[layer.name]
                grads[layer.name] = [layer_grads[0]] + [j-i for i, j in zip(layer_grads[:-1], layer_grads[1:])]
            for param in self._params:
                param_grads = grads[param.name]
                grads[param.name] = [param_grads[0]] + [j-i for i, j in zip(param_grads[:-1], param_grads[1:])]

        # Reset the layers to their `free state' values, where they were initially
        for layer, state in zip(self._layers, layers_free): layer.state = state
        
        return grads

    def _standard_param_grads(self, layers_first, layers_second):
        """Compute the parameter gradients as gradients of free-clamped power difference

        Args:
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            param_grads: list of Tensors. The parameter gradients
        """

        # Compute the cost gradients of the free state
        # for layer, free in zip(self._layers, layers_free): layer.state = free
        # grads_free = [updater.grad() for updater in self._param_updaters_cost]

        # Compute the energy gradients of the first state
        for layer in self._layers: layer.state = layers_first[layer.name]
        grads_free = [updater.grad() for updater in self._param_updaters]

        # Compute the energy gradients of the second (clamped) state
        for layer in self._layers: layer.state = layers_second[layer.name]
        grads_clamped = [updater.grad() for updater in self._param_updaters]

        # Compute the parameter gradients
        eta = self._second_nudging - self._first_nudging
        param_grads = [(clamped - free) / eta for free, clamped in zip(grads_free, grads_clamped)]   # In training.monitor.Optimizer we subclass torch.optim.SGD, with maximize=False, meaning the parameter updates are given by param -> param - gradient. Therefore to have \dot{k} \propto \partial_k (P_F - P_C) we need gradient = - \partial_k (P_F - P_C)

        return param_grads

    def _layer_grads(self, layers_first, layers_second):
        """Compute the layer gradients given the activations of the first and second states

        Args:
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            layer_grads: list of Tensors. The layer gradients
        """

        # FIXME: the gradients of the output layer are wrong because we need to set the correct 'output force' (nudging) at each time step.

        # Compute the energy gradients of the first state
        for layer in self._layers: layer.state = layers_first[layer.name]
        grads_first = [updater.grad() for updater in self._layer_updaters]

        # Compute the energy gradients of the second state
        for layer in self._layers: layer.state = layers_second[layer.name]
        grads_second = [updater.grad() for updater in self._layer_updaters]

        # Compute the layer gradients
        batch_size = self._layers[0].state.size(0)
        layer_grads = [(second - first) / ((self._second_nudging - self._first_nudging) * batch_size) for first, second in zip(grads_first, grads_second)]

        return layer_grads

    def _set_nudgings(self):
        """Sets the nudging values of the first and second states, depending on the attributes variant and nudging

        first_nudging: nudging value used to compute the first state of equilibrium propagation
        second_nudging: nudging value used to compute the second state of equilibrium propagation
        """

        if self._variant == "positive":
            self._first_nudging = 0.
            self._second_nudging = self._nudging
        elif self._variant == "negative":
            self._first_nudging = -self._nudging
            self._second_nudging = 0.
        elif self._variant == "centered":
            self._first_nudging = -self._nudging
            self._second_nudging = self._nudging
        else:
            # TODO: this raise of error is redundant
            raise ValueError("expected 'positive', 'negative' or 'centered' but got {}".format(self._variant))

    def __str__(self):
        formula = 'alternative' if self._use_alternative_formula else 'standard'
        return 'Equilibrium propagation -- mode={}, nudging={}, formula={}'.format(self._variant, self._nudging, formula)
