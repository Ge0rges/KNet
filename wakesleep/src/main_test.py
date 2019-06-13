import mnist_loader
import network
import numpy as np
from PIL import Image


def show_image(array, title):
    array = ((array * 256).astype(np.uint8).reshape([28, 28]))

    img = Image.fromarray(array)
    img.save(title + ".png")
    img.show(title + ".png")


def training_loop():
    training_data, validation_data, test_data = mnist_loader.load_only_inputs()

    total_data = training_data + validation_data + test_data

    net = network.WakeSleep([784, 30, 5], cost=network.QuadraticCost)

    test_data = zip(total_data, total_data)

    print "Total Cost: ", net.total_cost(test_data)

    for i in xrange(5):
        print "Stage", i + 1
        for j in xrange(5 - i):
            print "\tWake Phase ", j + 1
            net.wake_phase(total_data, i + 1, (i + 1) * 100, .5 / pow(10, i))
            print "\tSleep Phase", j + 1
            net.sleep_phase(i + 1, (i + 1) * 100, .1 / pow(10, i))
            print "Total Cost: ", net.total_cost(test_data)

    input_obj = np.random.randn(5, 1)
    for i in xrange(5):
        for index in xrange(len(input_obj)):
            if index == i:
                input_obj[index] = 1
            else:
                input_obj[index] = 0

        output = net.generate(input_obj)

        output = ((output * 256).astype(np.uint8).reshape([28, 28]))

        img = Image.fromarray(output)
        img.save("Number" + str(i + 1) + ".png")

    return net


def test_loop(net):
    input_obj = np.random.randn(5, 1)
    numbers = raw_input("Enter 5 Inputs: ")

    while numbers != "QUIT":
        numbers = numbers.split(" ")
        for i in range(5):
            input_obj[i] = float(numbers[i])
        print input_obj

        output = net.generate(input_obj)
        for i in range(28):
            for j in range(28):
                print output[i * 28 + j],
            print ""

        show_image(output, "Image")

        numbers = raw_input("Enter 5 Inputs: ")

if __name__ == '__main__':
    net = training_loop()
    test_loop(net)
