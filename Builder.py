import BasicElement
from random import randint, sample


# TODO определение subnet в отдельном модуле
class LQVSubnet:
    __slots__ = ['_synapses', 'neurons']

    def __init__(self, synapses, neurons):
        self._synapses = synapses
        self.neurons = neurons

    def run(self, number_of_iterations=1, reset_state=False):
        pass


def randomized_lqv_subnet_builder(
        sensors,

        number_of_neurons,
        number_of_synapses,

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

    neurons = [
        BasicElement.LQVNeuron(dendrites=sample(dendrites, number_of_dendrites),
                               number_of_kernels=randint(min_number_of_kernels, max_number_of_kernels))
        for number_of_dendrites in rnd(number_of_neurons, min_number_of_dendrites, max_number_of_dendrites)]

    signal_sources = sensors + neurons

    # создаем синапсы
    # распределяем синапсы по дендритам
    # возвращаем LQV-subnet
