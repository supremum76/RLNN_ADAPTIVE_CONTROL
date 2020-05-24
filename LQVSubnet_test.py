from Builder import randomized_lqv_subnet_builder


class Sensor(list):
    def get_output_signal(self, signal_index):
        return self[signal_index]

    def get_number_of_outputs(self):
        return len(self)


# class patterns
# 0 0 1 1
# 1 1 0 0
# 1 0 1 0
# 0 1 0 1
sensor = Sensor([0.0, 0.0, 0.0, 0.0])

lqv_subnet = randomized_lqv_subnet_builder(
    sensors=[sensor],

    number_of_neurons=10,

    min_number_of_synapses=3,
    max_number_of_synapses=5,

    min_number_of_dendrites=2,
    max_number_of_dendrites=5,

    min_number_of_kernels=2,
    max_number_of_kernels=5)

err = 0
n = 10000
for i in range(n):
    cls_num = i % 4
    if cls_num == 0:
        sensor[0] = 1
        sensor[1] = 1
        sensor[2] = 0
        sensor[3] = 0
    elif cls_num == 1:
        sensor[0] = 0
        sensor[1] = 0
        sensor[2] = 1
        sensor[3] = 1
    elif cls_num == 2:
        sensor[0] = 1
        sensor[1] = 0
        sensor[2] = 1
        sensor[3] = 0
    else:
        sensor[0] = 0
        sensor[1] = 1
        sensor[2] = 0
        sensor[3] = 1

    lqv_subnet.run(number_of_iterations=40, reset_state=False)

    # A trained LQV-neuron must set up a related kernel vector for each class pattern.
    # The signal value correlate with the distance from the class pattern to the corresponding kernel vector.
    avg_min_signal = 0
    for neuron in lqv_subnet.neurons:
        min_signal = 1.0
        for k in range(neuron.get_number_of_outputs()):
            min_signal = min(min_signal, neuron.get_output_signal(k))
        avg_min_signal += min_signal

    avg_min_signal /= len(lqv_subnet.neurons)

    if avg_min_signal > 0.2:
        err += 1

print("error, % = ", err * 100 / n)
