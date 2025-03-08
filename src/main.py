import numpy as np
import math
import random


def activation_func(num) -> float:
    return (1 / (1 + math.exp(num)))


class Layer:
    """
    Layer

    Members:

    Methods:
    
    """
    def __init__(self, n = 1):
        self.vector = np.array([Nueron() for _ in range(n)])


class Nueron:
    """
    Nueron

    Members:

    Methods:

    """

    def __init__(self):
        self.prev_layer = None
        self.activation = 0.0
        self.weights = []
        self.bias = 0.0 

    def set_weights(self):
        if self.prev_layer:
            pass

    def set_bias(self):
        if self.prev_layer:
            pass

    def randomize_weights(self):
        if self.prev_layer:
            for i in range(len(self.prev_layer.vector)):
                self.weights[i] = float(random.randrange(-100,100) / 100)

    def randomize_bias(self):
        if self.prev_layer:
            self.bias = float(random.randrange(-100,100) / 100)

    def frame_update_for_nueron_weights(self): 
        self.weights = [0 for i in range(len(self.prev_layer.vector))]

    # Only for hideen layers
    def set_activation(self):
        if not self.prev_layer:
            for i in range(len(self.prev_layer.vector)):
                pre_bias = self.prev_layer.vector[i].activation * self.weights[i]
            self.activation = activation_func(pre_bias + self.bias)


class Network():
    """
    Network

    Members:

    Methods:
    
    """
    def __init__(self, n_layered : int):
        self.n_layered = n_layered
        self.layer_list = [Layer() for _ in range(n_layered)]
        self.input_layer = self.layer_list[0]
        self.output_layer = self.layer_list[-1] 
    
    def set_input_layer_size(self,n):
        self.layer_list[0] = Layer(n)
        self.frame_update()
    
    def set_output_layer_size(self,n):
        self.layer_list[-1] = Layer(n)
        self.frame_update()
    
    def set_hidden_layer_size(self,n):
        for i in range(1,self.n_layered - 1):
            self.layer_list[i] = Layer(n)
        for i in range(1, len(self.layer_list)):
            for j in range(len(self.layer_list[i].vector)):
                self.layer_list[i].vector[j].prev_layer = self.layer_list[i - 1]

    def randomize_weights(self):
        for i in range(1,len(self.layer_list)):
            for j in range(len(self.layer_list[i].vector)):
                self.layer_list[i].vector[j].frame_update_for_nueron_weights()  
                self.layer_list[i].vector[j].randomize_weights()

    def randomize_bias(self):
        for i in range(1,len(self.layer_list)):
            for j in range(len(self.layer_list[i].vector)): 
                self.layer_list[i].vector[j].randomize_bias()


    def debug(self):
        for i in range(self.n_layered):
            print(f"\n__LAYER__ [{i + 1}]\n")
            for j in range(len(self.layer_list[i].vector)):
                # print(f" _nueron_{j + 1} : {self.layer_list[i].vector[j].activation} w/ bias {self.layer_list[i].vector[j].bias} w/ previous layer : {self.layer_list[i].vector[j].prev_layer}")
                print(f" _nueron_{j + 1} : {self.layer_list[i].vector[j].activation} w/ bias {self.layer_list[i].vector[j].bias} w/ weigths : {self.layer_list[i].vector[j].weights}")


    def print_layers(self):
        for i in range(self.n_layered):
            print(f"\n__LAYER__ [{i + 1}]\n")
            for j in range(len(self.layer_list[i].vector)):
                print(f" _nueron_{j + 1} : {self.layer_list[i].vector[j].activation} w bias {self.layer_list[i].vector[j].bias}")

    def frame_update(self):
        self.input_layer = self.layer_list[0]
        self.output_layer = self.layer_list[-1]



class Akira():
    pass


def main():
    network = Network(4)
    network.set_input_layer_size(10)
    network.set_output_layer_size(2)
    network.set_hidden_layer_size(5)
    network.randomize_weights()
    network.randomize_bias()
    network.debug()



if __name__ == "__main__":
    main()

# Randomly Generate weights and biases
# Cost Function (Study Cost Function) Since this takes in al weights
# WE LOCAL MINIMUM to reduce cost function, local minimum functions, what happens to the activations, overshooting
# negative gradient of the cost function 

# Understand the cost function more
# Back Propogation

# the activations are continous since the cost functions needs to be continuous to be able to find local mins
# Gradient Descent



# Lets write a network with random weights and biases

