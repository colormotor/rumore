
# Table of Contents

1.  [Vectorized value & gradient noise with NumPy](#org538e10d)
    1.  [Install with PIP](#orgbf7e820)
    2.  [Install locally](#orga324b4b)
    3.  [Examples](#org3b350da)



<a id="org538e10d"></a>

# Vectorized value & gradient noise with NumPy

Rumore is a lightweight Python library for procedural noise. It provides **value noise** and **gradient noise** in **1D/2D/3D** plus **octave summation** of these (fractal Brownian motion, fBm).

The library does not implement Ken Perlin’s original algorithm directly, but its gradient noise produces visually similar results.


<a id="orgbf7e820"></a>

## Install with PIP

    pip install rumore


<a id="orga324b4b"></a>

## Install locally

Clone the repo, navigate to the directory and from there

    pip install -e .


<a id="org3b350da"></a>

## Examples

Import necessary stuff

    import matplotlib.pyplot as plt
    import numpy as np
    import rumore
    
    rumore.set_defaults()

Generate some 1d gradient noise with different number of octaves

    x = np.linspace(-10, 10, 200)
    plt.figure(figsize=(5,4))
    for i in range(1, 8):
        plt.plot(x, rumore.grad_noise(x, octaves=i))
    plt.show()

![img](https://raw.githubusercontent.com/colormotor/rumore/main/figures/1d.png)

Perturb points along a circle with 2d noise

    t = np.linspace(0, np.pi*2, 200)
    x = np.cos(t)
    y = np.sin(t)
    r = rumore.grad_noise(x, y)*0.5+0.5
    plt.figure(figsize=(4,4))
    plt.plot(x*r, y*r)
    plt.show()

![img](https://raw.githubusercontent.com/colormotor/rumore/main/figures/2d.png)

Generate a grid of 2d noise value (a grayscale image)

    x = np.linspace(0, 5, 300)
    y = np.linspace(0, 3, 100)
    img = rumore.noise_grid(x, y)
    plt.imshow(img)
    plt.show()

![img](https://raw.githubusercontent.com/colormotor/rumore/main/figures/2d_grid.png)

Set noise properties. Here we set the `octave_map` property, defining a function that is applied to the noise of each octave.
Note that in a notebook these states are persistent, which can give unexpected results. You can reset everything by using `rumore.set_defaults()`.

    rumore.cfg.lacunarity = 2.0
    rumore.cfg.falloff = 0.5 # falloff for each octave (persistence)
    rumore.cfg.set_degree(5) # Interpolation degree (3 or 5)
    rumore.cfg.shift = 0 #120.321 # Shift between octaves
    rumore.cfg.octave_map = lambda x: np.sin(x*np.pi*4)
    
    x = np.linspace(0, 5, 300)
    y = np.linspace(0, 3, 100)
    img = rumore.noise_grid(x, y)
    plt.imshow(img, cmap='gray')
    plt.show()

![img](https://raw.githubusercontent.com/colormotor/rumore/main/figures/2d_grid_2.png)

