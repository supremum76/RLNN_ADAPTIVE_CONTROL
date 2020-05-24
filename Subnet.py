import BasicElement


class LQVSubnet:
    __slots__ = ['_synapses', 'neurons']

    def __init__(self, synapses, neurons):
        self._synapses = synapses
        self.neurons = neurons

    def run(self, number_of_iterations=1, reset_state=False):
        if reset_state:
            for neuron in self.neurons:
                neuron.reset_state()

        for _ in range(number_of_iterations + 1):
            for synapse in self._synapses:
                synapse.run()

            for neuron in self.neurons:
                neuron.run()


class DSubnetLayer:
    __slots__ = ['_synapses', 'neurons']

    def __init__(self, synapses, neurons):
        self._synapses = synapses
        self.neurons = neurons

    def run(self, reward, reset_state=False):
        if reset_state:
            for neuron in self.neurons:
                neuron.reset_state()

        for synapse in self._synapses:
            synapse.run()

        for neuron in self.neurons:
            neuron.run(reward)


class DSubnet:
    __slots__ = ['_layers', 'output_synapses']

    def __init__(self, layers):
        self._layers = layers
        self.output_synapses = tuple(BasicElement.Synapse(neuron, 0) for neuron in layers[len(layers) - 1].neurons)

    def run(self, reward, reset_state=False):
        for layer in self._layers:
            layer.run(reward, reset_state)

        for synapse in self.output_synapses:
            synapse.run()