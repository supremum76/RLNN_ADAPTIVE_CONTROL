import random
import BasicElement


class Sensor(list):
    def get_output_signal(self, signal_index):
        return self[signal_index]


# class patterns
# 0 0 1 1
# 1 1 0 0
# 1 0 1 0
# 0 1 0 1
sensor = Sensor([0.0, 0.0, 0.0, 0.0])

synapses = [BasicElement.Synapse(sensor, 0),
            BasicElement.Synapse(sensor, 1),
            BasicElement.Synapse(sensor, 2),
            BasicElement.Synapse(sensor, 3)]

dendrites = [BasicElement.Dendrite([synapses[i]]) for i in range(len(synapses))]

neuron = BasicElement.LQVNeuron(dendrites, number_of_kernels=3)

err = 0
n = 5000
for i in range(n):
    rnd = random.randint(0, 2)
    if rnd == 0:
        sensor[0] = 1
        sensor[1] = 1
        sensor[2] = 0
        sensor[3] = 0
    elif rnd == 1:
        sensor[0] = 0
        sensor[1] = 0
        sensor[2] = 1
        sensor[3] = 1
    elif rnd == 2:
        sensor[0] = 1
        sensor[1] = 0
        sensor[2] = 1
        sensor[3] = 0
    else:
        sensor[0] = 0
        sensor[1] = 1
        sensor[2] = 0
        sensor[3] = 1

    for s in synapses:
        s.run()

    neuron.run()

    # A trained LQV-neuron must set up a related kernel vector for each class pattern.
    # The signal value correlate with the distance from the class pattern to the corresponding kernel vector.
    min_signal = 1.0
    for k in range(neuron.get_number_of_outputs()):
        min_signal = min(min_signal, neuron.get_output_signal(k))

    if min_signal > 0.2:
        err += 1

    if i >= 100 and i % 100 == 0:
        neuron.lqv_kernels_reactivation(threshold_frequency=0.05)

print("error, % = ", err * 100 / n)
