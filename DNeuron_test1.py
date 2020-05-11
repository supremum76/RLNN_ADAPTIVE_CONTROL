import BasicElement
import random


class Sensors(list):
    def get_output_signal(self, signal_index):
        return self[signal_index]


# 0 0 1 1 -> 1
# 1 1 0 0 -> 1
# 1 0 1 0 -> 0
sensor = Sensors([0.0, 0.0, 0.0, 0.0])

synapses = [BasicElement.Synapse(sensor, 0),
            BasicElement.Synapse(sensor, 1),
            BasicElement.Synapse(sensor, 2),
            BasicElement.Synapse(sensor, 3)]

dendrites = [BasicElement.Dendrite(synapses[:2]), BasicElement.Dendrite(synapses[2:])]

neuron = BasicElement.DNeuron(dendrites, number_of_kernels=3)

err = 0
reward = 0
n = 1000
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
    else:
        sensor[0] = 1
        sensor[1] = 0
        sensor[2] = 1
        sensor[3] = 0

    for s in synapses:
        s.run()

    neuron.run(reward)

    output_signal = neuron.get_output_signal()
    if (rnd in [0, 1] and output_signal != 1) or (rnd == 2 and output_signal == 1):
        err += 1
        reward = -1
    else:
        reward = 1

    if i >= 100 and i % 100 == 0:
        neuron.lqv_kernels_reactivation(threshold_frequency=0.05)

print("error, % = ", err * 100 / n)