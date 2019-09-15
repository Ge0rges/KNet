import mnist_loader
import network2
import numpy as np
import os
from PIL import Image

script_dir = os.path.dirname(__file__)


def show_image(array, title):
    array = ((array * 256).astype(np.uint8).reshape([28, 28]))

    img = Image.fromarray(array)
    img.save(title + ".png")
    img.show(title + ".png")


def training_loop(in_size, out_size):
    training_data, validation_data, test_data = mnist_loader.load_only_inputs()

    total_data = training_data + validation_data + test_data

    total_data = total_data[100]

    net = network2.WakeSleep(in_size, out_size)

    test_data = zip(total_data, total_data)

    training_length = 6

    for i in range(training_length):
        print("Stage", i + 1)
        for j in range(training_length - i):
            print("\tWake Phase ", j + 1)
            net.wake_phase(total_data, i + 1)
            print("\tSleep Phase", j + 1)
            net.sleep_phase(i + 1)

        samples_gen(net, i + 1)

    return net


def samples_gen(net, stage, sample_size=20):
    dirName = "Stage " + str(stage)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    input_obj = np.random.randn(net.inner_size, 1)
    for i in xrange(net.inner_size):
        for j in xrange(sample_size + 1):
            for index in xrange(len(input_obj)):
                if index == i:
                    input_obj[index] = (1.0/sample_size) * j
                else:
                    input_obj[index] = 0

            output = net.generate(input_obj)

            output = ((output * 256).astype(np.uint8).reshape([28, 28]))

            img = Image.fromarray(output)
            image_size = 28 * 5, 28 * 5
            img = img.resize(image_size)
            path = script_dir + "/" + dirName + "/Number" + str(i) + "-" + str(j) + ".png"
            img.save(path, "png")


def test_loop(net):
    input_obj = np.random.randn(net.inner_size, 1)
    numbers = raw_input("Enter " + str(net.inner_size) + " Inputs: ")

    while numbers != "QUIT":
        numbers = numbers.split(" ")
        for i in range(net.inner_size):
            input_obj[i] = float(numbers[i])
        print(input_obj)

        output = net.generate(input_obj)
        for i in range(28):
            for j in range(28):
                print(output[i * 28 + j]),
            print("")

        show_image(output, "Image")

        numbers = raw_input("Enter " + str(net.inner_size) + " Inputs: ")

if __name__ == '__main__':

    net = training_loop(784, 10)
    test_loop(net)
