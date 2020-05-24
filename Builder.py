from random import randint, sample

import BasicElement
import Subnet


def _randomized_d_subnet_layer_builder(
        signal_sources,

        number_of_neurons,

        min_number_of_synapses,
        max_number_of_synapses,

        min_number_of_dendrites,
        max_number_of_dendrites,

        min_number_of_kernels,
        max_number_of_kernels,

        recurrent_computing=True):
    #  a series of n integer random numbers in the range [a, b]
    def rnd(n, a, b):
        for _ in range(n):
            yield randint(a, b)

    # Creating a pool of dendrites in sufficient quantities.
    # Synapses in dendrites will be added later.
    dendrites = [BasicElement.Dendrite()
                 for _ in range((min_number_of_dendrites + max_number_of_dendrites) * number_of_neurons // 2)]

    neurons = [BasicElement.DNeuron(dendrites=sample(dendrites, number_of_dendrites),
                                    number_of_kernels=randint(min_number_of_kernels, max_number_of_kernels))
               for number_of_dendrites in rnd(number_of_neurons, min_number_of_dendrites, max_number_of_dendrites)]

    if recurrent_computing:
        _signal_sources = signal_sources + neurons
    else:
        _signal_sources = signal_sources

    # creating a complete set of synapses for every possible signal source
    synapses = [BasicElement.Synapse(signal_source=signal_source, signal_index=signal_index)
                for signal_source in _signal_sources for signal_index in range(signal_source.get_number_of_outputs())]
    # synapses distribution for dendrites
    used_synapses = set()
    for dendrite in dendrites:
        subset_synapses = sample(synapses, randint(min_number_of_synapses, max_number_of_synapses))
        dendrite.append_synapses(subset_synapses)
        used_synapses.update(subset_synapses)
    # only used synapses remain
    synapses = list(used_synapses)
    # create and return D-subnet layer
    return Subnet.DSubnetLayer(synapses, neurons)


def randomized_d_subnet_builder(
        lqv_subnet,

        number_of_layers,
        number_of_neurons,

        min_number_of_synapses,
        max_number_of_synapses,

        min_number_of_dendrites,
        max_number_of_dendrites,

        min_number_of_kernels,
        max_number_of_kernels,

        recurrent_computing):
    layers = []
    for i in range(number_of_layers):
        layers.append(_randomized_d_subnet_layer_builder(
            lqv_subnet.neurons if i == 0 else layers[i - 1].neurons,

            number_of_neurons,

            min_number_of_synapses,
            max_number_of_synapses,

            min_number_of_dendrites,
            max_number_of_dendrites,

            min_number_of_kernels,
            max_number_of_kernels,

            recurrent_computing))

    return Subnet.DSubnet(layers)


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

    # Creating a pool of dendrites in sufficient quantities.
    # Synapses in dendrites will be added later.
    dendrites = [BasicElement.Dendrite()
                 for _ in range((min_number_of_dendrites + max_number_of_dendrites) * number_of_neurons // 2)]

    neurons = [BasicElement.LQVNeuron(dendrites=sample(dendrites, number_of_dendrites),
                                      number_of_kernels=randint(min_number_of_kernels, max_number_of_kernels))
               for number_of_dendrites in rnd(number_of_neurons, min_number_of_dendrites, max_number_of_dendrites)]

    signal_sources = sensors + neurons

    # creating a complete set of synapses for every possible signal source
    synapses = [BasicElement.Synapse(signal_source=signal_source, signal_index=signal_index)
                for signal_source in signal_sources for signal_index in range(signal_source.get_number_of_outputs())]
    # synapses distribution for dendrites
    used_synapses = set()
    for dendrite in dendrites:
        subset_synapses = sample(synapses, randint(min_number_of_synapses, max_number_of_synapses))
        dendrite.append_synapses(subset_synapses)
        used_synapses.update(subset_synapses)
    # only used synapses remain
    synapses = list(used_synapses)
    # create and return LQV-subnet
    return Subnet.LQVSubnet(synapses, neurons)
