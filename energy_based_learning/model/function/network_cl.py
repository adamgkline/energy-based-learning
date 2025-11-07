import copy
import torch

class TwinnedNetwork():
    def __init__(self, function):
        """Creates an instance of Network.

        Args:
            function (Function): the function used as an energy function
        """

        self._function = function

        self._input_layer = function.layers()[0]
        self._output_layer = function.layers()[-1]
        self._hidden_layers = function.layers()[1:-1]
        
        # initialize in "free state"
        self.unclamp()

    def free_layers(self):
        """Return the list of free layers"""
        return self._free_layers

    def clamp(self):
        self._is_clamped = True
        self._constrained_layers = [self._input_layer, self._output_layer]
        self._free_layers = self._hidden_layers

    def unclamp(self):
        self._is_clamped = False
        self._constrained_layers = [self._input_layer]
        self._free_layers = [self._output_layer] + self._hidden_layers

    def set_input(self, input_values, reset=False):
        """Set the input layer to input values

        Args:
            input_values: input image. Tensor of shape (batch_size, channels, width, height). Type is float32.
            reset (bool, optional): if True, resets the state of the network to zero.
        """

        old_batch_size = self._input_layer.state.size(0)
        batch_size = input_values.size(0)
        
        # we set the input tensor on the network's device
        self._input_layer.set_input(input_values.to(self._function._device))  # FIXME

        # we set the state of the network to zero if reset is True, or if the size of the batch of examples is different from the previous batch size.
        if reset or batch_size != old_batch_size:
            for layer in self._free_layers:
                layer.init_state(batch_size, self._function._device)  # FIXME