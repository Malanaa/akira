import numpy as np

class Layer:
    """
    Layer
    """
    def __init__(self, n):
        self.vector = np.array([Nueron()] * n)

class Nueron:
    """
    Nueron
    """
    def __init__(self, prev_layer : Layer = None):
        self.prev_layer = prev_layer
        self.activation = 0 
        self.type = "trivial_layer" if prev_layer == None else "non_trivial_layer"
        self.weights = [0] * len(prev_layer.vector) if len(prev_layer.vector > 0) else []
        self.bias = 0 if prev_layer != None else None
    def set_weights():
        pass
    def set_bias():
        pass
    # Only for hideen layers
    def set_activation(self):
        for i in range(len(self.prev_layer.vector)):
            pre_bias = self.prev_layer.vector[i].activation * self.weights[i]
        self.activation = pre_bias + self.bias

# layer_input = Layer(784)
# for i in range(len(layer_input.vector)):
#     print(f"{i}:{layer_input.vector[i].activation}")