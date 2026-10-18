import numpy as np
import gzip, pickle, json
import time

def print_img(matrix, max=255):
  for array in matrix:
    line = ""
    for saturation in array:
      value = saturation / max
      if value > 0.9:
        line += "@"
      elif value > 0.8:
        line += "%"
      elif value > 0.7:
        line += "#"
      elif value > 0.6:
        line += "*"
      elif value > 0.5:
        line += "+"
      elif value > 0.4:
        line += "="
      elif value > 0.3:
        line += "-"
      elif value > 0.2:
        line += ":"
      elif value > 0.1:
        line += "."
      else:
        line += " "
    print(line)

def load_data(filepath):
  with gzip.open(filepath, 'rb') as f:
    return pickle.load(f, encoding='bytes')

def save_network(filepath, masterList):
  with open(filepath, 'w') as file:
    json.dump(masterList, file)

def load_network(filepath):
  with open(filepath, 'r') as file:
    return json.load(file)




def transform(inputs):
  output = np.zeros_like(inputs)
  THRESHOLD = 0.9
  for i in range(len(inputs)):
    for j in range(len(inputs[i])):
      for k in range(len(inputs[i][j])):
        value = inputs[i][j][k]
        if value/255 > THRESHOLD:
          output[i][j][k] = 255.0
        else:
          output[i][j][k] = 0.0
  return output



# (train_x, train_y), (test_x, test_y) = load_data('mnist.pkl.gz')

# print(train_y)

# unique, counts = np.unique(train_y, return_counts=True)

# print(dict(zip(unique, counts)))






# # new_train = transform(train_x[:1000])

# i = 0
# while True:
#   time.sleep(0.7)
#   print_img(train_x[i])
#   print(train_y[i])
#   i+=1
