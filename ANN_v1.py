from utility import *

import numpy as np
import random

class NeuralLayer:
  def __init__(self, layer_size: int, prev_layer_size: int) -> None:
    self.size: int = layer_size
    self.prev_layer_size: int = prev_layer_size

    # self.weights: np.ndarray = 1*np.random.rand(self.size, self.prev_layer_size) # Edge weight matrix
    self.biases: np.ndarray = np.zeros(self.size) # Node bias vector

    self.weights: np.ndarray = (2*np.random.rand(self.size, self.prev_layer_size)) - (np.ones((self.size, self.prev_layer_size)))  # Edge weight matrix
    # self.biases: np.ndarray = (2*np.random.rand(self.size)) - (np.ones(self.size)) # Node bias vector


    self.activation_func: function = NeuralLayer.sigmoid
    self.activation_func_prime: function = NeuralLayer.sigmoid_prime

    self.outputs = None
    self.z_values = None

    self.d_outputs = None
    self.d_weights = None
    self.d_biases = None
    self.d_inputs = None
    self.d_z = None

    self.avg_d_weights = 0
    self.avg_d_biases = 0


  # TODO Need to formalize
  # Input: ndarray input vector of length self.prev_layer_size (output of prev layer)
  # Output: ndarray output vector of length self.size (input for next layer), also stores it in self.outputs
  def compute(self, input):
    # Copy input vector
    vec = np.copy(input)

    # Apply weights and biases to input vector
    vec = (self.weights @ input) + self.biases

    self.z_values = np.copy(vec)

    # Apply activation function
    new_vec = np.zeros_like(vec)
    for i in range(len(vec)):
      new_vec[i] = self.activation_func(vec[i])
    
    self.outputs = new_vec
    return new_vec
  
  def reset_derivatives(self):
    self.d_outputs = None
    self.d_weights = None
    self.d_biases = None
    self.d_inputs = None
    self.d_z = None

    self.avg_d_weights = 0
    self.avg_d_biases = 0

  # TODO UJFJDSKLFJKLDJSFKLJ
  # Input: ndarray vectors of the outputs of the prev layer and the derivatives of the next layer
  # Output: stores ndarray vector of derivatives in self.derivatives and returns it
  def backprop(self, prev_layer_outputs, training_set_size):
    d_biases = np.zeros(self.size)
    d_z = np.zeros(self.size)
    d_weights = np.zeros_like(self.weights)
    d_inputs = np.zeros(self.prev_layer_size)

    # Loops through neurons in current layer
    for i in range(self.size):
      dz = self.activation_func_prime(self.z_values[i]) * self.d_outputs[i]
      d_z[i] = dz
      d_biases[i] = 1 * dz

      # Loops through connected neurons in previous layer
      for j in range(len(prev_layer_outputs)):
        dweight = prev_layer_outputs[j] * dz
        d_weights[i,j] = dweight
    

    d_inputs = d_z @ self.weights

    self.d_biases = d_biases
    self.d_weights = d_weights
    self.d_inputs = d_inputs
    self.d_z = d_z

    self.avg_d_weights += ((1/training_set_size) * self.d_weights)
    self.avg_d_biases += ((1/training_set_size) * self.d_biases)

    return d_inputs


  # TODO k:FDOPKS:KFOKDS:FKLS:
  # Take 1 step towards greatest downward gradient
  def take_gradient_step(self, scaler):
    self.biases = self.biases - (scaler * self.avg_d_biases)
    self.weights = self.weights - (scaler * self.avg_d_weights)


  @staticmethod
  def sigmoid(x):
    LIMIT = 1000
    if x > LIMIT:
      return 1
    if x < -LIMIT:
      return 0
    
    return 1.0 / (1.0 + np.exp(-x))

  @staticmethod
  def sigmoid_prime(x):
    return NeuralLayer.sigmoid(x) * (1 - NeuralLayer.sigmoid(x))

  @staticmethod
  def ReLU(x):
    if x > 0:
      return x
    else:
      return 0
  
  @staticmethod
  def ReLU_prime(x):
    if x > 0:
      return 1
    else:
      return 0

  @staticmethod
  def test(x):
    return x
  
  @staticmethod
  def test_prime(x):
    return 1


class NeuralNetwork:

  # Input: tuple of layer sizes
  # Ex: (5, 4, 4, 3) = size-5 input layer, 2 size-4 hidden layers, size-3 output layer
  def __init__(self, shape: tuple[int]) -> None:
    self.shape: tuple[int] = shape

    # List of layers, input layer is not included (0th item is first hidden layer)
    self.layers: list[NeuralLayer] = []

    # Initialize layers
    for i in range(len(shape)-1):
      layer = NeuralLayer(shape[i+1], shape[i])
      self.layers.append(layer)

  # TODO Need to formalize
  # Input: ndarray input vector
  # Output: ndarray output vector
  def solve(self, input):
    # Copy input vector
    vec = np.copy(input)

    # Loop through and compute each layer
    for i in range(len(self.shape)-1):
      vec = self.layers[i].compute(vec)
    
    return vec
  
  def reset_derivatives(self):
    for layer in self.layers:
      layer.reset_derivatives()

  # Input: list of input vectors, list of target vectors (amount of inputs & targets must equal)
  # Output: net cost
  def calculate_total_cost(self, inputs, targets):
    sample_size = len(inputs)
    total_cost = 0

    for i in range(sample_size):
      prediction_vec = self.solve(inputs[i].flatten())
      target_vec = targets[i].flatten()
      single_cost = NeuralNetwork.calculate_single_cost(prediction_vec, target_vec)

      total_cost += single_cost

    return total_cost
  
  def calculate_avg_cost(self, inputs, targets):
    sample_size = len(inputs)

    total_cost = self.calculate_total_cost(inputs, targets)

    avg_cost = total_cost / sample_size

    return avg_cost

  
  def backprop_single(self, input, target, training_set_size=1):

    output = self.solve(input)
    dcost = 2 * (output - target)

    self.layers[-1].d_outputs = np.copy(dcost)

    for i in range(len(self.layers)-1, -1, -1):
      if i == 0:
        self.layers[i].backprop(input, training_set_size)
      else:
        prev_layer_outputs = self.layers[i-1].outputs
        prev_layer_d_output = self.layers[i].backprop(prev_layer_outputs, training_set_size)
        self.layers[i-1].d_outputs = prev_layer_d_output

  def backprop_set(self, inputs, targets):
    self.reset_derivatives()

    sample_size = len(inputs)
    for i in range(sample_size):
      self.backprop_single(inputs[i].flatten(), targets[i].flatten(), sample_size)

  def take_gradient_step(self, scaler):
    for layer in self.layers:
      layer.take_gradient_step(scaler)

  def save(self):
    master_list = []
    for layer in self.layers:
      master_list.append(layer.weights.tolist())
      master_list.append(layer.biases.tolist())
    
    return master_list

  def load(self, master_list):
    for layer in self.layers:
      layer.weights = np.array(master_list.pop(0))
      layer.biases = np.array(master_list.pop(0))

  # Input: output vector of NN, desired output vector (must be same length)
  # TODO Formalize
  @staticmethod
  def calculate_single_cost(prediction, target):
    cost = 0
    for i in range(len(prediction)):
      cost += (prediction[i] - target[i])**2
    return cost


def print_progress_bar(percent_complete, bar_length=10):
  print(f"[{'='*(percent_complete*bar_length//100)}{' '*(bar_length-(percent_complete*bar_length//100))}] {percent_complete}%")

def test_correctness(network, inputs, targets):
  sample_size = len(inputs)

  seen_amounts = [0]*10
  correct_amounts = [0]*10

  max_six = 0

  for i in range(sample_size):
    answer = targets[i]
    prediction = network.solve(inputs[i].flatten())
    prediction_num = prediction.argmax()

    # if answer == 6:
    #   print_img(inputs[i])
    
    seen_amounts[answer] += 1
    
    if prediction_num == answer:
      correct_amounts[answer] += 1
  
  for i in range(len(seen_amounts)):
    print(f"{i}: {(100 * correct_amounts[i])//seen_amounts[i]}%")
  
  return seen_amounts, correct_amounts


# IMPORTANT: All training arrays are references, not copies
def get_sorted_training_sets(inputs, targets):
  sorted_train_list = [ [] for _ in range(10) ] # List of empty lists

  for i in range(len(inputs)):
    sorted_train_list[targets[i]].append(inputs[i])

  return sorted_train_list

# IMPORTANT: All training arrays are references, not copies
def get_randomized_training_set(sorted_training_sets, digit_frequency_list, size):
  training_set = []
  total_freq = np.sum(digit_frequency_list)

  for num in range(len(digit_frequency_list)):
    percent_freq = digit_frequency_list[num] / total_freq
    amt_to_add = int(np.round(percent_freq * size))

    for _ in range(amt_to_add):
      training_set.append([random.choice(sorted_training_sets[num]), num])
  
  np.random.shuffle(training_set)

  training_inputs, training_targets = tuple(zip(*training_set))

  return training_inputs, training_targets


def get_target_vecs(target_num_list):
  vec = np.zeros((len(target_num_list), 10))
  
  for i in range(len(target_num_list)):
    vec[i][target_num_list[i]] = 1
  
  return vec


# ====================================================================================================
# ========== Main Code ===============================================================================
# ====================================================================================================

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
LOAD_FILE = 'network-v1.1.2.json'
# SAVE_FILE = 'network-v1.1.3.json'


network = NeuralNetwork((784, 16, 16, 10))
network.load(load_network(LOAD_FILE))

(train_x, train_y), (test_x, test_y) = load_data('mnist.pkl.gz')
# train_x = train_x / 255.0 # scale pixel brightness to decimal
sorted_train_sets = get_sorted_training_sets(train_x, train_y)



#                     0    1    2    3    4    5    6    7    8    9
training_freq_list = [1.1, 1.0, 1.0, 1.4, 1.0, 1.5, 1.4, 1.0, 1.5, 1.2]



# =======================================================

test_correctness(network, train_x, train_y)



# TODO test training


# print('READY!')

# for img in train_x:
#   input()
#   print_img(img)
#   pred = network.solve(img.flatten())
#   print(pred)
#   print(pred.argmax())
#   print()
#   for i in range(len(pred)):
#     print(f"{i}: ", end='')
#     print_progress_bar(int(100*pred[i]), 20)









# sample_size = 5000

# train_targets = get_target_vecs(train_y)
# batches_x = np.split(train_x, len(train_x) // sample_size)
# batches_y = np.split(train_targets, len(train_targets) // sample_size)

# prev_cost = 0

# print(prev_cost)


# step_size = 0.05
# for i in range(750):
#   # if i % 10 == 0:
#   #   command = input()
#   #   if command == "save":
#   #     save_network(SAVE_FILE, network.save())
#   #     break
#   #   else:
#   #     step_size = float(command)
  
#   batch_x, batch_y = batches_x[i % len(batches_x)], batches_y[i % len(batches_y)]

#   network.backprop_set(batch_x, batch_y)
#   network.take_gradient_step(step_size)

#   print(i)
#   new_cost = network.calculate_avg_cost(batch_x, batch_y)
#   print(new_cost)
#   print(new_cost - prev_cost)
#   if (new_cost > prev_cost):
#     print("==================================================")
#   else:
#     print()
#   prev_cost = new_cost

# save_network(SAVE_FILE, network.save())











# # sample_size = 5000

# # prev_cost = 0

# # print(prev_cost)

# # step_size = 0.03
# # for i in range(400):
# #   # if i % 10 == 0:
# #   #   command = input()
# #   #   if command == "save":
# #   #     save_network(SAVE_FILE, network.save())
# #   #     break
# #   #   else:
# #   #     step_size = float(command)
  
# #   rand_train_x, rand_train_y = get_randomized_training_set(sorted_train_sets, training_freq_list, sample_size)

# #   rand_train_targets = get_target_vecs(rand_train_y)
  
# #   network.backprop_set(rand_train_x, rand_train_targets)
# #   network.take_gradient_step(step_size)

# #   print(i)
# #   new_cost = network.calculate_avg_cost(rand_train_x, rand_train_targets)
# #   print(new_cost)
# #   print(new_cost - prev_cost)
# #   if (new_cost > prev_cost):
# #     print("==================================================")
# #   else:
# #     print()
# #   prev_cost = new_cost

# # save_network(SAVE_FILE, network.save())


