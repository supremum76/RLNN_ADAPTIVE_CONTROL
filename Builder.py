import BasicElement
from random import randint, sample


# TODO определение subnet в отдельном модуле
# TODO отдельный модуль Template для класса SignalSource
class LQVSubnet:
    __slots__ = ['_synapses', 'neurons']

    def __init__(self, synapses, neurons):
        self._synapses = synapses
        self.neurons = neurons

    def run(self, number_of_iterations=1, reset_state=False):
        pass


class DSubnetLayer:
    __slots__ = ['_synapses', 'neurons']

    def __init__(self, synapses, neurons):
        self._synapses = synapses
        self.neurons = neurons

    def run(self, reward, reset_state=False):
        if reset_state:
            for neuron in self.neurons:
                # TODO реализовать в нейронах метод reset_state
                neuron.reset_state()

        for synapse in self._synapses:
            synapse.run()

        for neuron in self.neurons:
            neuron.run(reward)


class DSubnet:
    __slots__ = ['_layers', 'output_layer']

    def __init__(self, layers):
        self._layers = layers
        self.output_layer = tuple(layers[len(layers)])

    def run(self, reward, reset_state=False):
        for layer in self._layers:
            layer.run(reward, reset_state)


def randomized_lqv_subnet_builder(
        sensors,

        number_of_neurons,

        min_number_of_synapses,
        max_number_of_synapses,

        min_number_of_dendrites,
        max_number_of_dendrites,

        min_number_of_kernels,
        max_number_of_kernels):
    #  a series of n integer random numbers in the range [a, b]
    def rnd(n, a, b):
        for _ in range(n):
            yield randint(a, b)

    # Создаем пул дендритов в достаточном количестве.
    # Синапсы в дендриты будут добавлены позже.
    dendrites = [BasicElement.Dendrite()
                 for _ in range((min_number_of_dendrites + max_number_of_dendrites) * number_of_neurons // 2)]

    neurons = [BasicElement.LQVNeuron(dendrites=sample(dendrites, number_of_dendrites),
                                      number_of_kernels=randint(min_number_of_kernels, max_number_of_kernels))
               for number_of_dendrites in rnd(number_of_neurons, min_number_of_dendrites, max_number_of_dendrites)]

    signal_sources = sensors + neurons

    # создаем полный набор синапсов для каждого возможного сигнала
    synapses = [BasicElement.Synapse(signal_source=signal_source, signal_index=signal_index)
                for signal_source in signal_sources for signal_index in range(signal_source.get_number_of_outputs())]
    # распределяем синапсы по дендритам
    used_synapses = set()
    for dendrite in dendrites:
        subset_synapses = sample(synapses, randint(min_number_of_synapses, max_number_of_synapses))
        dendrite.append_synapses(subset_synapses)
        used_synapses.update(subset_synapses)
    # оставляем только используемые синапсы
    synapses = list(used_synapses)
    # возвращаем LQV-subnet
    return LQVSubnet(synapses, neurons)
