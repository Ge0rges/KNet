import mnist_loader
import network
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_only_inputs()

total_data = training_data + validation_data + test_data

net = network.WakeSleep([784, 30, 5], cost=network.QuadraticCost)

test_data = zip(total_data, total_data)

print "Total Cost: ", net.total_cost(test_data)

for i in xrange(5):
    print "Stage", i + 1
    for j in xrange(5 - i):
        print "\tWake Phase ", j + 1
        net.wake_phase(total_data, i + 1, (i + 1) * 100, 5.0 / (i * 5 + 1))
        print "\tSleep Phase", j + 1
        net.sleep_phase(i + 1, (i + 1) * 100, 1.0 / (i * 5 + 1))
        print "Total Cost: ", net.total_cost(test_data)


input_obj = np.random.randn(5, 1)
numbers = raw_input("Enter 5 Inputs: ")

while numbers != "QUIT":
    numbers = numbers.split(" ")
    for i in range(5):
        input_obj[i] = float(numbers[i])
    print input_obj

    output = net.generate(input_obj)
    for i in range(32):
        for j in range(32):
            print output[i][j],
        print ""

    numbers = raw_input("Enter 5 Inputs: ")
