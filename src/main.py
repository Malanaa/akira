import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import json



class Layer:
    """
    Layer

    Members:

    Methods:
    
    """
    def __init__(self, n = 1):
        self.vector = np.array([Nueron() for _ in range(n)])


def sigmoid_squish(num) -> float:
    '''
    Sigmoid
    '''
    return (1 / (1 + math.exp(num)))


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

    def set_weights(self, listweights):
        if self.prev_layer:
            self.weights = listweights

    def set_bias(self, bias):
        if self.prev_layer:
            self.bias = bias
            

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
        if self.prev_layer:
            for i in range(len(self.prev_layer.vector)):
                pre_bias = self.prev_layer.vector[i].activation * self.weights[i]
            self.activation = sigmoid_squish(pre_bias + self.bias)

    def return_kernel(self):
        n = int(math.sqrt(len(self.weights)))
        ker = np.array(self.weights)
        return ker.reshape((n,n))



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

    def set_input_layer_values(self, input_activations):
        if(len(self.layer_list[0].vector) != len(input_activations)):
            # add exception handling
            # Doesnt work because input values size need to be equal to size of layer
            return 
        for i in range(len(input_activations)):
            self.layer_list[0].vector[i].activation = input_activations[i]

    
    def set_output_layer_size(self,n):
        self.layer_list[-1] = Layer(n)
        self.frame_update()
    
    # Also sets previous layers for every neuron
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


    def set_all_activation(self):
        for i in range(1,len(self.layer_list)):
            for j in range(len(self.layer_list[i].vector)):
                self.layer_list[i].vector[j].set_activation()


    def debug(self):
        for i in range(self.n_layered):
            print(f"\n__LAYER__ [{i + 1}]\n")
            for j in range(len(self.layer_list[i].vector)):
                # print(f" _nueron_{} : {self.layer_list[i].vector[j].activation} w/ bias {self.layer_list[i].vector[j].bias} w/ previous layer : {self.layer_list[i].vector[j].prev_layer}")
                print(f" _nueron_{j} : {self.layer_list[i].vector[j].activation} w/ bias {self.layer_list[i].vector[j].bias} w/ weigths : {self.layer_list[i].vector[j].weights}")


    def print_layers(self):
        for i in range(self.n_layered):
            print(f"\n__LAYER__ [{i + 1}]\n")
            for j in range(len(self.layer_list[i].vector)):
                print(f" _nueron_{j} : {self.layer_list[i].vector[j].activation} w bias {self.layer_list[i].vector[j].bias}")

    def frame_update(self):
        self.input_layer = self.layer_list[0]
        self.output_layer = self.layer_list[-1]
    
    def prediction_last_layer(self):
        ls = []
        for i in range(len(self.layer_list[-1].vector)):
            ls.append(self.layer_list[-1].vector[i].activation)
        return max(ls), ls.index(max(ls))
    

    def return_all_kernals(self):
        kernals = {}
        # excluding the input layer
        for i in range(1, len(self.layer_list)):
            for j in range(len(self.layer_list[i].vector)):
                kernals[f' kernal __layer__{i} __nueron__{j}'] = self.layer_list[i].vector[j].return_kernel()
        return kernals


class NetworkWrite:

    def __init__(self, network : Network):
        '''
        Network must already have all weights and biases set.
        '''
        self.network = network
        self.data = {}
        self.metadata = {}
        self.total_parameters = 0
    
    def fill_data(self, name : str):

            for i in range(1, len(self.network.layer_list)):
                for j in range(len(self.network.layer_list[i].vector)):
                    self.total_parameters += (len(self.network.layer_list[i].vector[j].weights) + 1)

            self.metadata = {
                "network-name" : name, 
                'network-num-layers': len(self.network.layer_list),
                "network-birth" : datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "network-input-layer-size" : len(self.network.layer_list[0].vector),
                "network-output-layer-size" :len(self.network.layer_list[-1].vector),
                "network-hidden-layer-size" : len(self.network.layer_list[1].vector),
                "network-total-parameters" : self.total_parameters 
            }
        
            # excluding the input layer
            for i in range(1, len(self.network.layer_list)):
                for j in range(len(self.network.layer_list[i].vector)):
                    weights = []
                    weights = self.network.layer_list[i].vector[j].weights
                    self.data[f"l{i}n{j}"] = [weights,self.network.layer_list[i].vector[j].bias]

    
    def record_data(self, filename, model_name=f"network@time{datetime.datetime.now().strftime("%H:%M:%S")}"):
        self.fill_data(model_name)
        with open(filename, "w") as outfile:
            json.dump({"metadata": self.metadata, "data": self.data}, outfile, indent=4)


class NetworkRead:
    def __init__(self):
        self.network = None
        self.metadata = None
        self.personality = None

    def get_network_from(self,filename): 
        with open(filename, "r") as file:
            data =  json.load(file)
        
        self.metadata = data["metadata"] #  a dictionary
        self.personality = data['data'] #  a dictionary
        self.network = Network(self.metadata['network-num-layers'])
        self.network.set_input_layer_size(self.metadata['network-input-layer-size'])
        self.network.set_output_layer_size(self.metadata['network-output-layer-size'])
        self.network.set_hidden_layer_size(self.metadata['network-hidden-layer-size'])
        for i in range(1,len(self.network.layer_list)):
            for j in range(len(self.network.layer_list[i].vector)):
                self.network.layer_list[i].vector[j].weights = self.personality[f"l{i}n{j}"][0]
                self.network.layer_list[i].vector[j].bias = self.personality[f"l{i}n{j}"][1]

        return self.network


        # return self.network
        
    
    
def main():

    # Loading in the dala
    df = pd.read_csv('dataset_digits/train.csv')
    # labels
    label = df['label']
    data = df.drop('label', axis=1)

    # img i for example
    i = 1
    label_i = label.iloc[i]
    data_i = data.iloc[i]
    image_i = data.iloc[i].values.reshape(28,28)

    # raw 1D vector that i need
    raw_image_vector = data.iloc[i].values

    # plt.imshow(image_i, cmap='gray')
    # print(label_i)
    # plt.show()

    # set n-layered nueral network
    # network = Network(4)

    # # # First set size of Input and Output Layer
    # network.set_input_layer_size(784)
    # network.set_output_layer_size(10)

    # # # Setting values of the input layer
    # network.set_input_layer_values(raw_image_vector)


    '''
    The order here matters, you must initialze the first and last layer first. this is because of how the functions are called 
    . set_hidden_layer_size() sizes up the weights list for all the hidden layers and the last layer. 
    ( I could make it so that the last layer is done after but there is no point, if a usecase shows up ill change it)
    '''
    # then set size of hidden layers
    # network.set_hidden_layer_size(16)

    # # # Randomizing just to test
    # network.randomize_weights()
    # network.randomize_bias()


    # Set all activations dependant on the weights and biases
    # network.set_all_activation()

    # # debugging
    # network.debug()

    # # the end prediction
    # prediction_weight, number_predicted = network.prediction_last_layer()
    # print(f"predicted: {number_predicted} w/ activation {prediction_weight}")


    # # plot these out in matplot lib alr
    # all_kernals = network.return_all_kernals()

    # write_network = NetworkWrite(network=network)
    # write_network.record_data("network.json")

    # network_parser = NetworkRead()
    # new_network = network_parser.get_network_from('network.json')
    # new_network.set_all_activation()
    # new_network.set_input_layer_values(raw_image_vector)
    # new_network.debug()

    '''
    Now we can parse in a network from a json file :D
    '''
    # network_parser = NetworkRead()
    # new_network = network_parser.get_network_from('network.json')
    # new_network.set_all_activation()
    # new_network.set_input_layer_values(raw_image_vector)
    # new_network.debug()



    load_network = NetworkRead()
    load_network.assign_network('network.json')



if __name__ == "__main__":
    main()

# save and load weights and biases

# Find a way to save kernals(weights and biases for each neuron ) and then fetch them again save them as a json
# Learn MatPlotLib to bueatifully display how akira thinks
# Cost Function (Study Cost Function) Since this takes in al weights
# WE LOCAL MINIMUM to reduce cost function, local minimum functions, what happens to the activations, overshooting
# negative gradient of the cost function 

# Understand the cost function more
# Back Propogation

# the activations are continous since the cost functions needs to be continuous to be able to find local mins
# Gradient Descent

# Lets write a network with random weights and biases

