# lets do all our tests in this file, so that accessing folders is easy

# test out network
from CCN.cclayer import cclayer
from CCN.ccnet import ccnet
import wakesleep.src.mnist_loader as mnist_loader
import wakesleep.src.network2 as network2
import numpy as np
import os
from PIL import Image
import json

def ccn_test():

    n = 5
    x_data = [i for i in range(n)]
    y_data = [(i + 1) for i in range(n)]

    one_hot_x = np.zeros((n, n))
    one_hot_y = np.zeros((n, n))

    i = 0
    for x, y in zip(x_data, y_data):
        vec = np.zeros((n, ))
        vec[x] = 1.0
        one_hot_x[i] = vec
        vec[x] = 0.0
        if i == n - 1:
            break
        vec[y] = 1.0
        one_hot_y[i] = vec
        i += 1

    print(one_hot_x.shape)
    print(one_hot_y.shape)
    
    N = n
    M = n
    alpha = 1e-2
    epochs = 10
    loss_epsilon = 3.0
    network = ccnet(N, M, alpha, loss_epsilon)
    # print out shapes of output and hidden weights of network
    # for i, layer in enumerate(network.layers):
    #     o_w = layer.o_weights
    #     h_w = layer.h_weights
    #     print("o weight shape for layer {} => {}".format(i, o_w.shape))
    #     for h in h_w:
    #         print("h weight shape for layer {} => {}".format(i, h.shape))
    #     print("\n")

    # printint out shapes of cached outputs from forward pass
    # print("shabba")
    # cache = network.forward_pass(one_hot_x[0])
    # preds, hidden = cache['y'], cache['hidden']
    # for y in preds:
    #     print(y.shape)
    # print("hello")
    
    # for h in hidden:
    #     print(h)
    
    network.train(one_hot_x, one_hot_y, epochs)  
    
    
script_dir = os.path.dirname(__file__)
def show_image(array, title):
    array = ((array * 256).astype(np.uint8).reshape([28, 28]))

    img = Image.fromarray(array)
    img.save(title + ".png")
    img.show(title + ".png")


def training_loop(in_size, out_size):
    training_data, validation_data, test_data = mnist_loader.load_only_inputs()
    
    total_data = training_data + validation_data + test_data

    net = network2.WakeSleep(in_size, out_size)

    test_data = zip(total_data, total_data)

    training_length = 6

    for i in range(training_length):
        print("Stage", i + 1)
        for j in range(training_length - i):
            print("\tWake Phase ", j + 1)
            net.wake_phase(total_data, i + 1, (i + 1) * 100, 1.0 / pow(10, i))
            print("\tSleep Phase", j + 1)
            net.sleep_phase(i + 1, (i + 1) * 100, 1.0 / pow(10, i))

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
    for i in range(net.inner_size):
        for j in range(sample_size + 1):
            for index in range(len(input_obj)):
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
                print(output[i * 28 + j],)
            print("")

        show_image(output, "Image")

        numbers = raw_input("Enter " + str(net.inner_size) + " Inputs: ")

if __name__ == '__main__':

    # ws_ccn_test runnable
    # net = training_loop(784, 10)
    # test_loop(net)
    
    # ccn_test runnable
    # ccn_test()

    pass
