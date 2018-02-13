from matplotlib import mpl,pyplot
import numpy as np

# make values from -5 to 5, for this example
#world = "y  g  r  y  r  y  g  w  r  \
#r  g  y  g  y  y  g  w  g  \
#w  w  y  w  y  r  g  g  g  \
#y  y  y  r  g  g  g  r  y  \
#r  w  y  g  b  r  r  y  w  \
#r  g  w  y  g  r  g  y  w  \
#w  r  g  g  w  r  r  r  w  \
#y  w  r  g  r  r  r  w  y  \
#g  r  g  y  g  w  y  g  y"
world = "k  r  b  r  b  k  r  b  y  \
y  c  m  y  r  b  r  r  w  \
k  b  k  r  b  w  y  c  c  \
m  m  b  k  y  r  k  c  w  \
g  k  w  m  w  c  c  y  k  \
y  c  m  y  g  k  r  y  k  \
r  m  r  m  c  b  k  g  g  \
m  b  m  b  r  y  b  b  y  \
y  y  b  w  m  y  m  g  k"
zvals = []
zrow = []
count = 0
for c in world.split():
    if c == "w":
        zrow.append(0)
    elif c == "b":
        zrow.append(1)
    elif c == "r":
        zrow.append(2)
    elif c == "y":
        zrow.append(3)
    elif c == "g":
        zrow.append(4)
    elif c == "m":
        zrow.append(5)
    elif c == "k":
        zrow.append(6)
    elif c == "c":
        zrow.append(7)
    count += 1    
    if count % 9 == 0 and count >0:
        zvals.append(zrow[:])  
        zrow = [] 

print np.array(zvals)

# make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['white', 'blue','red','yellow','green','magenta', 'black', 'cyan'])
bounds=[0,1,2,3,4,5,6,7,8]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = pyplot.pcolor(np.array(zvals),
                    cmap = cmap,norm=norm, edgecolor='k', lw=2)
pyplot.axis('off')
pyplot.savefig("ExperimentSparseGridNoTerminalWorld.png")
  
#pyplot.grid()
pyplot.show()
