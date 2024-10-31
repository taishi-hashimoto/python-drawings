# %% This script shows what the flicker noise looks like
import numpy as np
from tqdm.auto import tqdm


class Flicker:
    """Flicker noise generator.
    
    Based on:
    - Barnes, Efficient Numerical and Analog Modeling of Flicker Noise Processes, 1971.  
      https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote604.pdf"""

    def __init__(self, v: np.ndarray = None):
        """Initializes the state.
        
        Parameters
        ==========
        v: floating point np.ndarray of the size (5, 2)"""
        if v is None:
            # SET INITIAL VALUES V(I, 1) TO ZERO MEAN VALUES
            v = np.zeros((5, 2))  # DIMENSION V(5, 2)
        self.v = np.array(v)
        assert self.v.shape == (5, 2)

    def __call__(self, size: int, dtype=None) -> np.ndarray:
        """Generate specified number of samples following flicker noise.
        
        Parameters
        ==========
        size: int
            The number of samples.
        dtype: type
            float or complex
        """
        # GENERATE AND PRINT N FLICKER NUMBERS
        if dtype == complex:
            size *= 2
        a = np.zeros(size)
        v = self.v
        # SELECT RANDOM V(1, 2) UNIFORMLY DISTRIBUTED ON (-1/2, 1/2)
        r = np.random.uniform(-1/2, 1/2, size=size)
        for i in range(size):
            v[0, 1] = r[i]
            # SOLVE RECURSION RELATIONS
            v[1, 1] = .999771 * v[1, 0] + .333333 * v[0, 1] - .333105 * v[0, 0]
            v[2, 1] = .997942 * v[2, 0] + .333333 * v[1, 1] - .331276 * v[1, 0]
            v[3, 1] = .981481 * v[3, 0] + .333333 * v[2, 1] - .314815 * v[2, 0]
            v[4, 1] = .833333 * v[4, 0] + .333333 * v[3, 1] - .166667 * v[3, 0]
            a[i] = v[4, 1]
            # RESET V
            v[:, 0] = v[:, 1]
        if dtype == complex:
            return a[0::2] + 1j*a[1::2]
        else:
            return a


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from os.path import expanduser
    from itertools import chain
    flicker = Flicker()
    noi = flicker(4096)
    
    plt.figure(figsize=(4, 4))
    plt.plot(noi)
    plt.tight_layout()
    plt.savefig(expanduser("flicker_noise_timeseries.png"))
    
    faxis = np.fft.rfftfreq(len(noi), d=1/20e6)
    spc = np.abs(np.fft.rfft(noi))
    plt.figure(figsize=(4, 4))
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(faxis, spc)
    plt.tight_layout()
    plt.savefig(expanduser("flicker_noise_spectrum.png"))
    
    # The number of samples averaged
    e = np.logspace(1, 6)
    
    samples = []
    p = 0
    with tqdm(e) as bar:
        for x in bar:
            k = int(x - p)
            y = flicker(k)
            samples.append(y)
            p += k
    means = [np.mean(list(chain.from_iterable(samples[:i]))) for i, _ in enumerate(e, 1)]
    # Average of stdandard normal
    means2 = [np.mean(np.random.normal(size=int(x))) for x in e]
    
    c0 = "C0"
    c1 = "C1"
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax2 = ax.twinx()
    ax.plot(e, means, color=c0)
    ymin, ymax = ax.get_ylim()
    yoff = max(abs(ymin), abs(ymax))
    ax.set_ylim(-yoff, yoff)
    ax.set_xscale("log")
    ax.set_ylabel("Flicker noise", color=c0)
    ax.tick_params(axis="y", which="both", colors=c0)
    ax2.plot(e, means2, color=c1)
    ymin, ymax = ax2.get_ylim()
    yoff = max(abs(ymin), abs(ymax))
    ax2.set_ylim(-yoff, yoff)
    ax2.set_xscale("log")
    ax2.set_ylabel("Standard Normal", color=c1)
    ax2.tick_params(axis="y", which="both", colors=c1)
    ax.set_xlabel("Number of samples averaged")
    fig.tight_layout()
    fig.savefig(expanduser("flicker_noise_vs_normal.png"))
    plt.show()
    
# %%
