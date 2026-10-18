import numpy as np
import math

# v0.1 of the neural network with layers within the NeuralNetwork class

class NeuralNetwork:

  # Input: tuple of layer sizes
  # Ex: (5, 4, 4, 3) = size-5 input layer, 2 size-4 hidden layers, size-3 output layer
  def __init__(self, shape: tuple[int]) -> None:
    self.shape: tuple[int] = shape

    self.weights: list[np.ndarray] = [] # List of edge weight tables
    self.biases: list[np.ndarray] = [] # List of edge bias constant vectors

    self.activation_funcs = [NeuralNetwork.sigmoid] * (len(shape)-1)

    # Initialize weight matrices to zero
    for i in range(len(shape)-1):
      weight_matrix = np.zeros((shape[i+1], shape[i]))
      self.weights.append(weight_matrix)

    # Initialize bias vectors to zero
    for i in range(len(shape)-1):
      bias_vector = np.zeros(shape[i+1])
      self.biases.append(bias_vector)

  # TODO Need to formalize
  # Input: ndarray input vector
  # Output: ndarray output vector
  def solve(self, input):
    # Copy input vector
    vec = np.copy(input)

    # Loop through layers, computing each layer
    for i in range(len(self.shape)-1):
      # Perform matrix-vector multi and add bias vector
      vec = (self.weights[i] @ vec) + self.biases[i]

      # Apply activation function to result
      new_vec = np.zeros_like(vec)
      for j in range(len(vec)):
        new_vec[j] = self.activation_funcs[i](vec[j])
      
      vec = new_vec
    
    return vec
  
  # Input: output vector of NN, desired output vector (must be same length)
  # TODO Formalize
  @staticmethod
  def calculate_single_loss(prediction, target):
    loss = 0
    for i in range(len(prediction)):
      loss += (prediction[i] - target[i])**2
    return loss


  @staticmethod
  def sigmoid(x):
    return 1 / (1 + math.e**(-x))
  
  @staticmethod
  def ReLU(x):
    if x < 0:
      return 0
    else:
      return x

