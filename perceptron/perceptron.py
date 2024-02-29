import random
import numpy as np

class Perceptron():

    def __init__(self, number_of_inputs):
        self.NUMBER_OF_INPUTS = number_of_inputs
        # Initialize the weights with random values
        # self.w = np.random.random([number_of_inputs+1,1]) #weights + bias
        self.w = np.zeros([number_of_inputs+1,1])
        self.n = 1 # Learning rate
    
    def update_learning_rate(self, n:float):
        self.n = n

class PerceptronRosenblatt(Perceptron):

    def __init__(self, number_of_inputs):
        super().__init__(number_of_inputs)

    def activation_function(self, a:float):
        return 1 if a >= 0 else -1

    def calculate_output(self, u):
        """
        Calculate the output of the perceptron for a given input
        u: input vector

        return: the output of the perceptron      
        """
        # Calculate the dot product of the inputs and the weights
        activation = (np.transpose(self.w) @ u)[0][0]
        # Apply the activation function
        return self.activation_function(activation)

    def train_for_single_sample(self, u, y):
        """
        Train the perceptron for a single example
        u: input vector
        y: desired output
        """
        # Calculate the output of the perceptron for current weights
        x = self.calculate_output(u)
        # Update the weights
        self.w = self.w + self.n * (y - x) * u

    def print_hyperplane_function(self):
        for i in range(len(self.w)-1):
            print(f"{self.w[i][0]:0.3f}w{i}", end=" + ")
        print(f"{self.w[-1][0]:0.3f} = 0")

    def generate_linearly_space_separated_hyperplane(self):
        """
        Generate a string representation of the hyperplane function
        """
        hyperplane = ""
        for i in range(len(self.w)-1):
            hyperplane += f"{self.w[i][0]:0.3f}w{i} + "
        hyperplane += f"{self.w[-1][0]:0.3f} = 0"
        return hyperplane
    
    def seperation_line_function_for_2_input(self, weight:float)->float:
        w0 = self.w[0][0]
        w1 = self.w[1][0]
        bias = self.w[2][0]

        # w0*weight + w1*height + bias = 0 
        # height = (-w0*weight - bias)/w1

        height =  lambda x: (-w0*x-bias)/w1
        return height(weight)
    
