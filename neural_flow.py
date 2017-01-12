##################################################################
#
# Andrew Baker
#
# File Name:  neural_flow.py
# Date:  01/06/17
#
# Description:  Contains a list of classes and functions used
#  to simulate a neural network.
#
#
##################################################################

from functools import reduce


class Neuron:
    def __init__(self, inbound_neurons=None, label=''):
        if inbound_neurons is None:
            inbound_neurons = []

        # An optional description of the neuron - most useful for outputs.
        self.label = label

        # Neurons from which this Node receives values
        self.inbound_neurons = inbound_neurons

        # Neurons to which this Node passes values
        self.outbound_neurons = []

        # A calculated value
        self.value = None

        # Add this node as an outbound node on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_neurons` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Backward propagation.

        Compute the gradient of the current Neuron with respect
        to the input neurons. The gradient of the loss with respect
        to the current Neuron should already be computed in the `gradients`
        attribute of the output neurons.
        """
        raise NotImplemented


class Input(Neuron):
    def __init__(self):
        self.gradients = {}
        # An Input Neuron has no inbound neurons,
        # so no need to pass anything to the Neuron instantiator
        Neuron.__init__(self)

    # NOTE: Input Neuron is the only Neuron where the value
    # may be passed as an argument to forward().
    #
    # All other Neuron implementations should get the value
    # of the previous neurons from self.inbound_neurons
    #
    # Example:
    # val0 = self.inbound_neurons[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

    def backward(self):
        # An Input Neuron has no inputs so we refer to our self
        # for the gradient
        self.gradients = {self: 0}
        for n in self.outbound_neurons:
            self.gradients[self] += n.gradients[self]


class Add(Neuron):
    def __init__(self, *inputs):
        Neuron.__init__(self, inputs)

    def forward(self):
        add_neurons = [n.value for n in self.inbound_neurons]
        self.value = sum(add_neurons)


class Mul(Neuron):
    def __init__(self, *inputs):
        Neuron.__init__(self, inputs)

    def forward(self):
        self.value = reduce(lambda x, y: x * y, [n.value for n in self.inbound_neurons])


class Linear(Neuron):
    def __init__(self, inputs, weights, bias):
        Neuron.__init__(self, inputs)

        # NOTE:  The weights and bias properties here are not
        # numbers, but rather references to other neurons.
        # The weight and bias values are stored within the
        # respective neurons.
        self.weights = weights
        self.bias = bias

    def forward(self):
        self.value = self.bias.value
        for w, x in zip(self.weights, self.inbound_neurons):
            self.value += w.value * x.value


def topological_sort(feed_dict):
    """
    Sort the neurons in topological order using Kahn's Algorithm.

    :param feed_dict: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to
     that Neuron.

    Returns a list of sorted neurons.
    """

    input_neurons = [n for n in feed_dict.keys()]

    g = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in g:
            g[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in g:
                g[m] = {'in': set(), 'out': set()}
            g[n]['out'].add(m)
            g[m]['in'].add(n)
            neurons.append(m)

    l = []
    s = set(input_neurons)
    while len(s) > 0:
        n = s.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        l.append(n)
        for m in n.outbound_neurons:
            g[n]['out'].remove(m)
            g[m]['in'].remove(n)
            # if no other incoming edges add to s
            if len(g[m]['in']) == 0:
                s.add(m)
    return l


def forward_pass(output_neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

    :param output_neuron: A Neuron in the graph, should be the output Neuron (have no outgoing edges).
    :param sorted_neurons: a topologically sorted list of neurons.

    Returns the output Neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_neuron.value
