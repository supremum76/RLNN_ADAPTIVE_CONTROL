from Builder import randomized_lqv_subnet_builder, randomized_d_subnet_builder


class Sensor(list):
    def get_output_signal(self, signal_index):
        return self[signal_index]

    def get_number_of_outputs(self):
        return len(self)


# class patterns -> class
# 0 0 1 1 -> 0
# 1 1 0 0 -> 1
# 1 0 1 0 -> 0
# 0 1 0 1 -> 1
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

d_subnet = randomized_d_subnet_builder(
    lqv_subnet=lqv_subnet,

    number_of_layers=1,
    number_of_neurons=1,

    min_number_of_synapses=1,
    max_number_of_synapses=1,

    min_number_of_dendrites=10,
    max_number_of_dendrites=10,

    min_number_of_kernels=2,
    max_number_of_kernels=2,

    recurrent_computing=False)

err = 0
reward = 0
n = 10000
for i in range(n):
    cls_num = i % 2
    pattern_num = i % 4
    if pattern_num == 0:
        sensor[0] = 1
        sensor[1] = 1
        sensor[2] = 0
        sensor[3] = 0
    elif pattern_num == 1:
        sensor[0] = 0
        sensor[1] = 0
        sensor[2] = 1
        sensor[3] = 1
    elif pattern_num == 2:
        sensor[0] = 1
        sensor[1] = 0
        sensor[2] = 1
        sensor[3] = 0
    else:
        sensor[0] = 0
        sensor[1] = 1
        sensor[2] = 0
        sensor[3] = 1

    lqv_subnet.run(number_of_iterations=50, reset_state=False)
    d_subnet.run(reward, reset_state=False)

    sum_signals = 0
    for k in range(len(d_subnet.output_layer.neurons)):
        neuron = d_subnet.output_layer.neurons[k]
        sum_signals += neuron.get_output_signal()

    if sum_signals > len(d_subnet.output_layer.neurons) // 2:
        recognition_cls_num = 1
    else:
        recognition_cls_num = 0

    if recognition_cls_num != cls_num:
        reward = -1
        err += 1
    else:
        reward = +1

    if i >= 1000 and i % 1000 == 0:
        for neuron in lqv_subnet.neurons:
            neuron.lqv_kernels_reactivation(threshold_frequency=0.05)

        for layer in d_subnet.layers:
            for neuron in layer.neurons:
                neuron.lqv_kernels_reactivation(threshold_frequency=0.05)

print("error, % = ", err * 100 / n)
