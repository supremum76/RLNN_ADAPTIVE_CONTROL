import math
from random import random
from copy import copy

synapse_learning_rate = 0.01
lqv_learning_rate = 0.01
d_neuron_learning_rate = 0.01
lqv_kernels_usage_threshold_frequency = 0.05
lqv_kernels_usage_frequency_learning_rate = 0.01
d_neuron_reassessment_prob = 0.001


class Synapse:
    __slots__ = ['_signal_source', '_signal_index', 'weight', 'output_signal']

    def __init__(self, signal_source, signal_index=0, weight=1.0):
        self._signal_source = signal_source
        self._signal_index = signal_index
        self.weight = weight
        self.output_signal = 0.0

    def run(self):
        signal = self._signal_source.get_output_signal(self._signal_index)

        # learning
        self.weight = (self.weight * (1.0 - synapse_learning_rate) +
                       synapse_learning_rate * abs(self.output_signal - signal))

        self.output_signal = signal


class Dendrite:
    __slots__ = ['_synapses', 'output_signal']

    def __init__(self, synapses=None):
        self._synapses = [] if synapses is None else list(synapses)
        self.output_signal = 0.0

    def append_synapses(self, synapses):
        self._synapses += synapses

    def run(self):
        if len(self._synapses) == 0:
            self.output_signal = 0.0
            return

        s = 0.0
        w = 0.0
        for syn in self._synapses:
            s += syn.output_signal * syn.weight
            w += syn.weight

        if not math.isclose(0.0, w):
            self.output_signal = s / w
        else:
            self.output_signal = 0.0


class _LQVKernel:
    __slots__ = ['_kernel']

    def __init__(self, dim):
        # See book "Neural Computing: Theory and Practice. Philip D. Wasserman. Van Nostrand Reinhold, 1989"
        # convex combination method
        self._kernel = [1.0 / math.sqrt(dim) for _ in range(dim)]

    def learning(self, vector):
        k = self._kernel
        for i in range(len(k)):
            k[i] = k[i] * (1.0 - lqv_learning_rate) + lqv_learning_rate * vector[i]

    def get_distance(self, vector):
        return math.dist(self._kernel, vector)

    def get_distance_between_kernels(self, kernel):
        return self.get_distance(kernel._kernel)

    def __copy__(self):
        clone = type(self)(dim=0)
        clone._kernel = self._kernel.copy()
        return clone


class LQVNeuron:
    __slots__ = ['dendrites', '_kernels', '_kernels_usage_frequency', '__vector', '_output_signal']

    def __init__(self, dendrites, number_of_kernels):
        self.dendrites = tuple(dendrites)
        self._kernels = [_LQVKernel(dim=len(dendrites)) for _ in range(number_of_kernels)]
        self._kernels_usage_frequency = [0.0 for _ in range(number_of_kernels)]
        self.__vector = [0.0 for _ in range(len(dendrites))]
        self._output_signal = [0.0 for _ in range(number_of_kernels)]

    def get_number_of_outputs(self):
        return len(self._output_signal)

    def get_output_signal(self, signal_index):
        return self._output_signal[signal_index]

    def run(self):
        for i in range(len(self.dendrites)):
            self.dendrites[i].run()
            self.__vector[i] = self.dendrites[i].output_signal

        _vector_normalization(self.__vector)

        min_dist = 0.0
        nearest_kernel_index = -1
        for i in range(len(self._kernels)):
            dist = self._kernels[i].get_distance(self.__vector)
            self._output_signal[i] = dist

            if nearest_kernel_index < 0 or dist < min_dist:
                nearest_kernel_index = i
                min_dist = dist

        _vector_normalization(self._output_signal)

        # LQV learning
        self._kernels[nearest_kernel_index].learning(self.__vector)
        for i in range(len(self._kernels)):
            if i != nearest_kernel_index:
                self._kernels_usage_frequency[i] = \
                    self._kernels_usage_frequency[i] * (1.0 - lqv_kernels_usage_frequency_learning_rate)
            else:
                self._kernels_usage_frequency[i] = \
                    (self._kernels_usage_frequency[i] * (1.0 - lqv_kernels_usage_frequency_learning_rate) +
                     lqv_kernels_usage_frequency_learning_rate)

        _lqv_kernels_reactivation(self._kernels, self._kernels_usage_frequency)

    def reset_state(self):
        for i in range(len(self._output_signal)):
            self._output_signal[i] = 0.0


class DNeuron:
    __slots__ = ['dendrites', '_kernels', '_kernels_usage_frequency', '__vector', '_output_signal', '_h', '_r0', '_r1']

    def __init__(self, dendrites, number_of_kernels):
        self.dendrites = tuple(dendrites)
        self._kernels = [_LQVKernel(dim=len(dendrites)) for _ in range(number_of_kernels)]
        self._kernels_usage_frequency = [0.0 for _ in range(number_of_kernels)]
        self.__vector = [0.0 for _ in range(len(dendrites))]
        self._h = [0 for _ in range(number_of_kernels)]  # LQV-clusters activity history
        self._r0 = [0.0 for _ in range(number_of_kernels)]  # expected reward with an output value of 0
        self._r1 = [0.0 for _ in range(number_of_kernels)]  # expected reward with an output value of 1
        self._output_signal = 0

    def get_number_of_outputs(self):
        return 1

    def get_output_signal(self, _dummy=0):
        return self._output_signal

    def run(self, reward):
        for i in range(len(self.dendrites)):
            self.dendrites[i].run()
            self.__vector[i] = self.dendrites[i].output_signal

        _vector_normalization(self.__vector)

        # LQV block -----------------------------------------------------------
        min_dist = 0.0
        nearest_cluster_index = -1
        for i in range(len(self._kernels)):
            dist = self._kernels[i].get_distance(self.__vector)

            if nearest_cluster_index < 0 or dist < min_dist:
                nearest_cluster_index = i
                min_dist = dist

        # LQV learning
        self._kernels[nearest_cluster_index].learning(self.__vector)
        # ---------------------------------------------------------------------

        # F block -------------------------------------------------------------
        self.__block_f(nearest_cluster_index, reward)
        # ---------------------------------------------------------------------

        for i in range(len(self._kernels)):
            if i != nearest_cluster_index:
                self._kernels_usage_frequency[i] = \
                    self._kernels_usage_frequency[i] * (1.0 - lqv_kernels_usage_frequency_learning_rate)
            else:
                self._kernels_usage_frequency[i] = \
                    (self._kernels_usage_frequency[i] * (1.0 - lqv_kernels_usage_frequency_learning_rate) +
                     lqv_kernels_usage_frequency_learning_rate)

        _lqv_kernels_reactivation(self._kernels, self._kernels_usage_frequency)

    def __block_f(self, cluster_index, reward, learning_rate=d_neuron_learning_rate,
                  reassessment_prob=d_neuron_reassessment_prob):
        h = self._h
        r0 = self._r0
        r1 = self._r1

        if not math.isclose(0.0, reward):
            reward /= sum(h)

            for i in range(len(h)):
                if r0[i] > r1[i]:
                    r = r0
                else:
                    r = r1

                r[i] = r[i] * (1.0 - learning_rate) + learning_rate * reward * h[i]  # reinforcement learning
                h[i] = 0  # clear LQV-clusters activity history

                if random() < reassessment_prob:
                    r0[i] = r1[i]

        h[cluster_index] += 1

        if r0[cluster_index] > r1[cluster_index]:
            self._output_signal = 0
        else:
            self._output_signal = 1

    def reset_state(self):
        self._output_signal = 0


#  ------------------------------------ utilities ---------------------------------------------------
def _vector_normalization(vector):
    norm = math.hypot(*vector)  # Euclidean norm
    if not math.isclose(0.0, norm):
        for i in range(len(vector)):
            vector[i] /= norm


# The reactivated kernel is replaced with a copy of the nearest core with a sufficient level of activity.
# This choice is designed to minimize the change in neuron behavior after reactivation.
def _lqv_kernels_reactivation(kernels, kernels_usage_frequency):
    for i in range(len(kernels)):
        if kernels_usage_frequency[i] < lqv_kernels_usage_threshold_frequency:
            min_dist = 0.0
            copy_kernel_index = -1
            for k in range(len(kernels)):
                if kernels_usage_frequency[k] > lqv_kernels_usage_threshold_frequency:
                    dist = kernels[k].get_distance_between_kernels(kernels[i])
                    if min_dist > dist or copy_kernel_index < 0:
                        min_dist = dist
                        copy_kernel_index = k

            if copy_kernel_index != -1:
                kernels[i] = copy(kernels[copy_kernel_index])
                kernels_usage_frequency[i] = kernels_usage_frequency[copy_kernel_index]
