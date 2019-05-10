import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

total_data = training_data + validation_data + test_data

net = network.WakeSleep([784, 30, 10])

for i in xrange(30):
    net.wake(total_data, 30 - i, (i + 1) * 100, 10.0 / (i + 1))
    net.sleep(30 - i, (i + 1) * 100, 10.0 / (i + 1))
