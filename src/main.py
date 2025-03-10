import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import json
# import scipy.special as scp

class Layer:
    """
    Layer(n) : size n layer.

    """
    def __init__(self, n = 1):
        self.vector = np.array([Nueron() for _ in range(n)])


def cost_one(prediction, expected):
    cost = 0
    for i in range(len(expected)):
        # cost += (prediction[i].activation - expected[i])**2
        cost += (prediction[i].activation - float(expected[i]))**2
    return cost


def sigmoid(num) -> float:
    '''
    sigmoid squisification

    '''
    if num < 0:
        return 1 - 1 / (1 + math.exp(num))
    return 1 / (1 + math.exp(-num))

def relu(num) -> float:
    '''
    relu
    '''
    return max(0,num)
class Nueron:

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
        pre_bias = 0
        if self.prev_layer:
            for i in range(len(self.prev_layer.vector)):
                pre_bias += self.prev_layer.vector[i].activation * self.weights[i]
            self.activation = sigmoid(pre_bias + self.bias)

    def return_kernel(self):
        n = int(math.sqrt(len(self.weights)))
        ker = np.array(self.weights)
        return ker.reshape((n,n))



class Network():

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
    
    def reset_activations(self):
        for i in range(len(self.layer_list)):
            for j in range(len(self.layer_list[i].vector)):
                self.layer_list[i].vector[j].activation = 0 

    def fire_given_label(self,given_input_list,label):
        compare_vector = [0 for i in range(len(self.layer_list[-1].vector))]
        compare_vector[label] = 1
        self.set_input_layer_values(given_input_list)
        self.set_all_activation()
        cost = cost_one(list(self.layer_list[-1].vector), compare_vector)
        return cost
    


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




def create_new_random_784_16_16_10_network():
    network = Network(4)
    network.set_input_layer_size(784)
    network.set_output_layer_size(10)
    network.set_hidden_layer_size(16)
    network.randomize_bias()
    network.randomize_weights()
    write_network = NetworkWrite(network)
    write_network.record_data('random_network.json')

    
def average_cost_on_N_dataset(network : Network,data,label):
    '''
    '''
    avg_cost = 0
    for i in range(len(data)):
        cost = network.fire_given_label(data.iloc[i].values, label.iloc[i])
        avg_cost += cost
        activation, prediction = network.prediction_last_layer()
        print(f"__ {(i/len(data))*100}% __ : {label.iloc[i]} : predicted : {prediction} - certainty : {activation} - cost : {cost}")
        # network.debug()
    return avg_cost / len(data)

def create_average_cost_function(network : Network, data, label):
    pass
    # lp_cost_func = []
    # for i in range(len(data)):
    #     cost = network.fire_given_label(data.iloc[i].values, label.iloc[i])
    #     activation_for_weights = []
    #     for j in range(len(network.layer_list)):
    #         activation_for_weights.append(layer.activation for layer in network.layer_list[j].vector)
    #     lp_cost_func.append(activation_for_weights)
    #     activation_for_weights = []
    #     lp_cost_func.append([1])
    #     activation, prediction = network.prediction_last_layer()
    #     print(f" adding_tcf __ {(i/len(data))*100}% __ : {label.iloc[i]} : predicted : {prediction} - certainty : {activation} - cost : {cost}")
    # print(f"lenght of lp cost function is {len(lp_cost_func)}")
    # return lp_cost_func

def find_average_acc(network : Network,data,label):
    '''
    '''
    accurate = 0
    for i in range(len(data)):
        cost = network.fire_given_label(data.iloc[i].values, label.iloc[i])
        activation, prediction = network.prediction_last_layer()
        correct = False
        if(prediction == label.iloc[i]):
            correct = True
            accurate+=1
        print(f"__ {(i/len(data))*100}% __ : {label.iloc[i]} : predicted : {prediction} - certainty : {activation} - cost : {cost}: pw___{correct}")
        # network.debug()
    return (accurate / len(data))*100

def main():

    '''
    Loading the data with pandas
    '''
    df = pd.read_csv('dataset_digits/train.csv')
    label = df['label'] # Labels in {0-9}
    data = df.drop('label', axis=1) # training data


    # '''
    # Data for image i raw data fetch with 
    # '''
    # i = 1
    # label_i = label.iloc[i]
    # data_i = data.iloc[i]
    # image_i = data.iloc[i].values.reshape(28,28)
    # raw_image_vector = data.iloc[i].values # Flattened 1D vector for IMG


    '''
    Show the image using matplotlib
    '''
    # plt.imshow(image_i, cmap='gray')
    # print(label_i)
    # plt.show()


    '''
    Manually Initialzing a 4-layered nueral network with random weight's and biases.
    '''
    # network = Network(4)
    # network.set_input_layer_size(784)
    # network.set_output_layer_size(10)
    # network.set_hidden_layer_size(16)
    
    '''
    setting input values here manually
    '''
    # network.set_input_layer_values(raw_image_vector)

    # network.randomize_weights()
    # network.randomize_bias()
    '''
    calculating the activations
    '''
    # network.set_all_activation()
    '''
    debug
    '''
    # network.debug()
    '''
    getting the prediction and kernals
    '''
    # prediction_weight, number_predicted = network.prediction_last_layer()
    # print(f"predicted: {number_predicted} w/ activation {prediction_weight}")
    # all_kernals = network.return_all_kernals()


    '''Writing Existing Network to a Json file'''
    # write_network = NetworkWrite(network=network)
    # write_network.record_data("network.json")

    '''Reading Network from Existing Json file'''
    # network_parser = NetworkRead()
    # new_network = network_parser.get_network_from('network.json')
    # new_network.set_all_activation()
    # new_network.set_input_layer_values(raw_image_vector)
    # new_network.debug()



    '''
    Data for image i raw data fetch with 
    '''
    i = 1
    label_i = label.iloc[i]
    data_i = data.iloc[i]
    image_i = data.iloc[i].values.reshape(28,28)
    raw_image_vector = data.iloc[i].values # Flattened 1D vector for IMG

    '''
    Random Network generator
    '''
    create_new_random_784_16_16_10_network()



    '''
    Load data and get average cost over n images.
    '''
    load_network = NetworkRead()
    new_network = load_network.get_network_from('random_network.json')
    '''
    just training and getting the average cost
    '''
    average_cost = average_cost_on_N_dataset(network=new_network, data=data[:100], label=label[:100])
    print(f"THE AVERAGE COST OF THE MODEL WAS: {average_cost}")

    '''
    Average Accuracy
    '''
    # average_acc = find_average_acc(network=new_network, data=data[:1000], label=label[:1000])
    # print(f"your model had accuracy {average_acc}%")

    # cost = new_network.fire_given_label(given_input_list=raw_image_vector, label=label_i)
    # new_network.debug()
    # prediction_weight, number_predicted = new_network.prediction_last_layer()
    # print(f"original_image = {label_i}. predicted: {number_predicted} w/ activation {prediction_weight} \ncost : {cost}")


    # export a average cost function and then minimize it


if __name__ == "__main__":
    main()


