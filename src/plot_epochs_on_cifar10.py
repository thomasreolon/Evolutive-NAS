import json, os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if 'cifar_results.txt' in os.listdir('.'):
    with open('cifar_results.txt', 'r') as fin:
        data = json.load(fin)
        scores = data['scores']
        besties = data['best_per_epoch']


##################################### PLOT TRENDING OF SCORES ############################################

topscore = [[], [], []]

scores = [np.array(x) for x in scores[:30]]

x = list(range(len(scores)))
for score_type, score in zip(range(3), ('NTK', 'Jacobian', 'Activations')):
    y_best, y_worst, y_med = [], [], []
    for epoch in scores:
        y_best.append(epoch[:, score_type].min())
        y_worst.append(epoch[:, score_type].max())
        y_med.append(np.median(epoch[:, score_type]))
    plt.plot(x,y_best,  c='mediumspringgreen', label=f'{score}: best')
    plt.plot(x,y_worst, c='deepskyblue',       label=f'{score}: median')
    plt.plot(x,y_med,   c='steelblue',         label=f'{score}: worst')
    if (score!='Activations'): plt.yscale('log')
    plt.legend()
    plt.show()


##################################### PLOT 3D VIEW OF EVOLUTION ############################################

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

best_offsprings_per_epoch = [b[1] for b in besties]

def get_colors(n):
  colors=[]
  poss = '56789abcdef'   # from greenwater to blue
  l = len(poss)
  r = int(l**2/n)
  for i in range(n):
    b,c = (i*r)//l, (i*r)%l
    c = f'#00{poss[b]}{poss[c]}9a'
    colors.append(c)
  return colors

colors = get_colors(len(best_offsprings_per_epoch))
labels = [f'epoch {i+1}' for i in range(len(best_offsprings_per_epoch))] 

sizes = []
max_activ = 0
for v in best_offsprings_per_epoch:
    max_activ = max(max_activ, -v[2]+20)
    tot=5
    for v2 in best_offsprings_per_epoch:
        tot+= int(v[0]<=v2[0])+int(v[1]<=v2[1])+int(v[2]<=v2[2])
    sizes.append(float(tot))

for i,( s, c, l) in enumerate(zip(best_offsprings_per_epoch, colors, labels)):
    x, y, z = [s[0]], [s[1]], [max_activ+s[2]]

    if i%int(len(best_offsprings_per_epoch)/3)==0:
        ax.scatter(x,y,z, s=sizes[i], c=c, label=l)
    else:
        ax.scatter(x,y,z, s=sizes[i], c=c)

ax.scatter([0],[0],[0], s=120, c='black')

ax.set_xlabel('NTK')
ax.set_ylabel('Jacob')
ax.set_zlabel('Regions')

plt.legend()
plt.show()


##################################### PLOT 3D ALL ############################################

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

best_offsprings_per_epoch = [b[1] for b in besties]

def get_colors(n):
  colors=[]
  poss = '56789abcdef'   # from greenwater to blue
  l = len(poss)
  r = int(l**2/n)
  for i in range(n):
    b,c = (i*r)//l, (i*r)%l
    c = f'#00{poss[b]}{poss[c]}9a'
    colors.append(c)
  return colors

colors = get_colors(len(best_offsprings_per_epoch))
labels = [f'epoch {i+1}' for i in range(len(best_offsprings_per_epoch))] 

sizes = []
max_activ = 0
for v in best_offsprings_per_epoch:
    max_activ = max(max_activ, -v[2]+20)
    tot=5
    for v2 in best_offsprings_per_epoch:
        tot+= int(v[0]<=v2[0])+int(v[1]<=v2[1])+int(v[2]<=v2[2])
    sizes.append(float(tot))

for i,( s, c, l) in enumerate(zip(scores, colors, labels)):
    x, y, z = [s[:,0], s[:,1], s[:,2]+max_activ]

    if i%int(len(best_offsprings_per_epoch)/3)==0:
        ax.scatter(x,y,z, s=sizes[i], c=c, label=l)
    else:
        ax.scatter(x,y,z, s=sizes[i], c=c)

ax.scatter([0],[0],[0], s=120, c='black')

ax.set_xlabel('NTK')
ax.set_ylabel('Jacob')
ax.set_zlabel('Regions')

plt.legend()
plt.show()

################################################################################

x = range(10)
y = range(10)

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,10))

for i, row in enumerate(ax):
    x0 = scores[0][:,i]   # --> first epoch
    xn = scores[-1][:,i]  # --> last epoch
    for j, col in enumerate(row):
        y0 = scores[0][:,j]   # --> first epoch
        yn = scores[-1][:,j]  # --> last epoch
        col.scatter(xn,yn, c='mediumspringgreen', label='last epoch')
        col.scatter(x0,y0, c='steelblue',         label='epoch 0')
        
        col.set_yscale(j!=2 and 'log' or 'linear')
        col.set_xscale(i!=2 and 'log' or 'linear')
        col.legend()

for i in range(3):
    scores = ['NTK', 'Jacob', 'Activ']
    plt.setp(ax[-1, i],  xlabel=scores[i])
    plt.setp(ax[i, 0], ylabel=scores[i])

plt.tick_params()
plt.show()