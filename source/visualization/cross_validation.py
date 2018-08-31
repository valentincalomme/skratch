import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

import numpy as np

k = 11

x = np.linspace(0, 1, k)
y = np.ones_like(x)


train_color = 'g'
test_color = 'b'

legend_dict = {'Train': train_color,
               'Test': test_color}

patchList = []
for key in legend_dict:
    data_key = patches.Patch(color=legend_dict[key], label=key)
    patchList.append(data_key)

ims = []

fig = plt.figure()

for fold in range(k - 1):

    ax = fig.add_subplot(111)

    boxes = []

    for i in range(k):

        if i == fold:
            facecolor = test_color
            label = "Test"

            rect = Rectangle((x[i], 0), x[1] - x[0], 1, label=label, edgecolor='black', facecolor=facecolor)
            test = rect
        else:
            facecolor = train_color
            label = "Train"

            rect = Rectangle((x[i], 0), x[1] - x[0], 1, label=label, edgecolor='black', facecolor=facecolor)
            train = rect

        boxes.append(rect)

        pc = PatchCollection([rect], facecolor=facecolor, alpha=0.3,
                             edgecolor='black')

        ax.add_patch(rect)

    ims.append(boxes)
    plt.xticks([])
    plt.yticks([])
    plt.legend(handles=patchList, loc="lower left", mode="expand", ncol=2, bbox_to_anchor=(0, 1.02, 1, 0.2))


anim = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=1)

plt.show()

anim.save("visualization/gif/cross_validation.mp4")