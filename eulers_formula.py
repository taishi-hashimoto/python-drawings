"Animation script to explain Euler's formula."
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D as _, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as anim
from matplotlib.colors import Normalize


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)



def e(t, T, k=1, sign=-1):
    omega = 2 * np.pi / T
    return np.exp(sign * 1j * k * omega * t)


t = np.arange(0, 1, 0.002)
T = 0.1
yy = [e(t, T, sign=1), e(t, T, sign=-1)]



fig, axes = plt.subplots(1, 2, figsize=(12, 7), subplot_kw=dict(projection="3d"))

arrowprops = dict(lw=2, color="k", arrowstyle="-|>", mutation_scale=20)
textprops = dict(fontsize=18)
titles = "$k > 0$", "$k < 0$"
cmap_names = "cool", "cool_r"

fig.suptitle("$e^{ik\omega_0 t} = \cos k\omega_0 t + i \sin k\omega_0 t$")

lines = []
segments = []
for y, ax, cmap_name, title in zip(yy, axes, cmap_names, titles):
    ax.add_artist(Arrow3D([0, 1], [0, 0], [0, 0], **arrowprops))
    ax.text(1, 0, 0, "$t$", "x", **textprops)
    ax.add_artist(Arrow3D([0, 0], [0, 1.5], [0, 0], **arrowprops))
    ax.text(0, 1.5, 0, "$\mathcal{R}$", "x", **textprops)
    ax.add_artist(Arrow3D([0, 0], [0, 0], [0, 1.5], **arrowprops))
    ax.text(0, 0, 1.5, "$\mathcal{I}$", "x", **textprops)

    X = t
    Y = y.real
    Z = y.imag

    # Dummy plot for nice box
    ax.plot(X, Y, Z, color="r", lw=2)[0].set_visible(False)

    points = np.c_[X, Y, Z].reshape(-1, 1, 3)
    segment = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segment, cmap=plt.get_cmap(cmap_name, 256), norm=Normalize(np.min(Y), np.max(Y)))
    line = ax.add_collection(lc)
    line.set_array(Y)
    line.set_linewidth(3)
    
    segments.append(segment)
    lines.append(line)

    ax.set_box_aspect([3, 1, 1])
    ax.set_axis_off()
    ax.set_title(title)

fig.tight_layout()


def update(i):
    for line, segment in zip(lines, segments):
        line.set_segments(segment[:i])

ani = anim.FuncAnimation(fig, update, len(t), interval=20, repeat=True)
ani.save(f"eulers_formula.gif", anim.PillowWriter(fps=30))
plt.show()
