from utility import *
from ANN_v1 import *

import numpy as np
import random

import tkinter as tk


# from ANN_v1 import *

is_mouse_down = False

def on_mouse_press(event):
  globals()['is_mouse_down'] = True
  on_mouse_motion(event)

def on_mouse_release(event):
  globals()['is_mouse_down'] = False

def on_mouse_motion(event):
  if not globals()['is_mouse_down']:
    return
  
  x, y = event.x, event.y
  if (not (0 < x < CANVAS_SIZE)) or (not (0 < y < CANVAS_SIZE)):
    globals()['is_mouse_down'] = False
    return

  grid_x, grid_y = get_grid_pos(x, y, 28, 28, CANVAS_SIZE, CANVAS_SIZE)

  cell = grid[grid_y][grid_x]
  canvas.itemconfig(cell, fill='black')

  matrix[grid_y][grid_x] = 1
  
  matrix[grid_y-1][grid_x] = max(0.5, matrix[grid_y-1][grid_x])
  matrix[grid_y+1][grid_x] = max(0.5, matrix[grid_y+1][grid_x])
  matrix[grid_y][grid_x+1] = max(0.5, matrix[grid_y][grid_x+1])
  matrix[grid_y][grid_x-1] = max(0.5, matrix[grid_y][grid_x-1])


def reset_matrix(grid):
  for row in grid:
    for cell in row:
      canvas.itemconfig(cell, fill="white")
  
  for i in range(len(matrix)):
    for j in range(len(matrix[i])):
      matrix[i][j] = 0

def on_submit():
  print("submitted")
  print_img(globals()["matrix"], 1)
  pred = network.solve(np.array(matrix).flatten())
  print("Neural Network's Guess: " + str(pred.argmax()))
  # print(pred)
  print()
  for i in range(len(pred)):
    print(f"{i}: ", end='')
    print_progress_bar(int(100*pred[i]), 20)


  reset_matrix(grid)



def get_grid_pos(pixel_x, pixel_y, ncells_x, ncells_y, grid_w, grid_h):
  cell_w = grid_w / ncells_x
  cell_h = grid_h / ncells_y
  return (int(pixel_x / cell_w), int(pixel_y / cell_h))


def create_grid(cell_amt = (28, 28), color = "white"):
  grid = [ [] for _ in range(cell_amt[1]) ]

  cell_size = (canvas.winfo_reqwidth() / cell_amt[0], canvas.winfo_reqheight() / cell_amt[1])
  for x in range(cell_amt[0]):
    for y in range(cell_amt[1]):
      x1, y1 = x * cell_size[0], y * cell_size[1]
      x2, y2 = (x + 1) * cell_size[0], (y + 1) * cell_size[1]
      cell = canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline = 'black')
      grid[y].append(cell)

  return grid


CANVAS_SIZE = 600

matrix = [ [0]*28 for _ in range(28) ]


# =========


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
LOAD_FILE = 'network-v1.1.2.json'
SAVE_FILE = 'network-save.json'


network = NeuralNetwork((784, 16, 16, 10))
network.load(load_network(LOAD_FILE))



# =========

root = tk.Tk()
root.title('ANN')

canvas = tk.Canvas(root, height=CANVAS_SIZE, width=CANVAS_SIZE, bg="white", bd=0)
canvas.pack()
submit_btn = tk.Button(root, text="SUBMIT", command=on_submit)
submit_btn.pack()

root.update()


canvas.bind("<Button-1>", on_mouse_press)
canvas.bind("<ButtonRelease-1>", on_mouse_release)
canvas.bind("<Motion>", on_mouse_motion)

grid = create_grid()

root.mainloop()
