import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_only_inputs()

total_data = training_data + validation_data + test_data

net = network.WakeSleep([784, 30, 10], cost=network.QuadraticCost)

for i in xrange(10):
    print "Stage", i + 1
    for j in xrange(10 - i):
        print "Wake Phase ", j + 1
        net.wake_phase(total_data, 1, (i + 1) * 100, 10.0 / (i + 1))
        print "Sleep Phase", j + 1
        net.sleep_phase(1, (i + 1) * 100, 10.0 / (i + 1))
