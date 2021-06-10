import json, os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if 'cifar_results.txt' in os.listdir('.'):
    with open('cifar_results.txt', 'r') as fin:
        data = json.load(fin)
        scores = data['scores']
        besties = data['best_per_epoch']


##################################### PLOT TRENDING OF SCORES ############################################

topscore = [[], [], []]
for epoch in scores:
    for score_type in range(3):
        best_fn_score = 1e30
        for x in epoch:
            best_fn_score = min(best_fn_score, x[score_type])
        topscore[score_type].append(best_fn_score)

x = list(range(len(scores)))

plt.plot(x,topscore[0], c='mediumspringgreen', label='NTK Score')
plt.legend()
plt.show()

plt.plot(x,topscore[1], c='deepskyblue', label='Jacobian Score')
plt.legend()
plt.show()

plt.plot(x,topscore[2], c='steelblue', label='Activations Score')
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
['mediumspringgreen', 'deepskyblue', 'steelblue']
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

