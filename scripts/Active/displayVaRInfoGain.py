import matplotlib.pyplot as plt
import numpy as np


def plot_arrow(state, width, ax, direction):
    h_length = 0.15
    shaft_length = 0.4
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state / width
    print x_coord, y_coord
    if direction is 'down':
        x_end = 0
        y_end = shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -shaft_length + h_length
    elif direction is 'left':
        x_end = -shaft_length + h_length
        y_end = 0
    elif direction is 'right':
        x_end = shaft_length - h_length
        y_end = 0
    else:
        print "ERROR: ", direction, " is not a valid action"
        return
    print x_end, y_end
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=0.2, head_length=h_length, fc='k', ec='k',linewidth=4) 

def plot_dot(state, width, ax):
    ax.plot(state % width, state / width, 'ko',markersize=10)

filePath = "/home/daniel/Code/PeARL_BIRL/data/active/"
filename_var = "var_maze.txt"
filename_pi = "pi_maze.txt"

#read VaR values for each initial state
f = open(filePath + filename_var)
mat = []
for line in f:
    row = []
    for el in line.strip().split(','):
        row.append(float(el))
    mat.append(row)


#read policy
#read VaR values for each initial state
f = open(filePath + filename_pi)
ax = plt.axes() 
pi = []
count = 0
rows = 0
for line in f:
    rows += 1
    row = []
    cols = len(line.strip().split(','))
    for el in line.strip().split(','):
        if el is "^":
            plot_arrow(count, cols, ax, "up")
        elif el is "v":
            plot_arrow(count, cols, ax, "down")
        elif el is ">":
            plot_arrow(count, cols, ax, "right")
        elif el is "<":
            plot_arrow(count, cols, ax, "left")
        elif el is ".":
            plot_dot(count, cols, ax)
        count += 1
        

heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
# Add the grid
ax = plt.gca()
# Minor ticks
ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
ax.grid(which='minor', axis='both', linestyle='-', linewidth=2, color='k')
#remove ticks
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off',
    right='off',
    labelbottom='off',
    labelleft='off') # labels along the bottom edge are off

cbar = plt.colorbar(heatmap)
cbar.ax.tick_params(labelsize=20) 
plt.show()
