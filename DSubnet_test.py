from Builder import randomized_lqv_subnet_builder, randomized_d_subnet_builder


# see Template.SignalSource
class Sensor(list):
    def get_output_signal(self, signal_index):
        return self[signal_index]

    def get_number_of_outputs(self):
        return len(self)


patterns = [
    [0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1]
]

sensor = Sensor([0, 0, 0, 0])

lqv_subnet = randomized_lqv_subnet_builder(
    sensors=[sensor],

    number_of_neurons=10,

    min_number_of_synapses=1,
    max_number_of_synapses=1,

    min_number_of_dendrites=10,
    max_number_of_dendrites=10,

    min_number_of_kernels=2,
    max_number_of_kernels=2)

d_subnet = randomized_d_subnet_builder(
    lqv_subnet=lqv_subnet,

    number_of_layers=1,
    number_of_neurons=30,

    min_number_of_synapses=1,
    max_number_of_synapses=1,

    min_number_of_dendrites=10,
    max_number_of_dendrites=10,

    min_number_of_kernels=10,
    max_number_of_kernels=10,

    recurrent_computing=False)

err = 0
reward = 0
n = 10000
for i in range(n):
    pattern_num = i % 4

    sensor[0] = patterns[pattern_num][0]
    sensor[1] = patterns[pattern_num][1]
    sensor[2] = patterns[pattern_num][2]
    sensor[3] = patterns[pattern_num][3]

    cls_num = patterns[pattern_num][4]

    # lqv_subnet -> d_subnet -> voting
    lqv_subnet.run(number_of_iterations=50, reset_state=False)
    d_subnet.run(reward, reset_state=False)

    # weighted voting
    sum_signal = 0
    sum_weight = 0
    for k in range(len(d_subnet.output_synapses)):
        synapse = d_subnet.output_synapses[k]
        sum_signal += synapse.signal * synapse.weight
        sum_weight += synapse.weight

    output_signal = sum_signal / sum_weight

    recognition_cls_num = round(output_signal)

    if recognition_cls_num == cls_num:
        reward = +1
    else:
        reward = -1
        err += 1

print("error, % = ", err * 100 / n)
