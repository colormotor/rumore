#!/usr/bin/env python3
import numpy as np
import numbers
import pdb

class Config:
    """
    Global configuration for noise generation.

    Attributes
    ----------
    seed : int
        Base seed for hashing / randomness (affects reproducibility).
    shift : float
        Coordinate shift applied at each octave in fBm.
    lacunarity : float
        Frequency multiplier per octave (controls detail).
    falloff : float
        Amplitude multiplier per octave (controls roughness).
    fade : Callable
        Interpolation function (cubic or quintic), set by set_degree().
    degree : int
        Current interpolation degree (3 or 5).
    """
    def __init__(self):
        self.seed = 0xF00D
        self.shift = 131.2322
        self.lacunarity = 2.0
        self.falloff = 0.5
        self.set_degree(5)  # default to quintic fade (Perlin-style, C² continuous)
        self.octave_map = lambda x: x # Modify to apply a function to each octave

    def set_degree(self, n):
        """
        Select the interpolation degree for smoothing between lattice points.

        Parameters
        ----------
        n : int
            Supported values:
            - 3 : cubic fade (t^2 * (3 - 2t)), C¹ continuous
            - 5 : quintic fade (t^3 * (t*(6t - 15) + 10)), C² continuous

        Raises
        ------
        ValueError
            If degree is not 3 or 5.
        """
        funcs = {
            3: lambda t: t*t*(3.0-2.0*t),
            5: lambda t: t * t * t * (t * (t * 6 - 15) + 10)
        }
        if n not in funcs:
            raise ValueError(f'Only degrees {list(funcs.keys())} are supported')
        self.fade = funcs[n]
        self.degree = n


cfg = Config()

def value_noise(*args, octaves=6):
    """
    Generate value noise in 1D, 2D, or 3D.

    Parameters
    ----------
    *args : float or ndarray
        Coordinates (x), (x, y), or (x, y, z). Each can be a scalar or numpy array.
    octaves : int, default=6
        Number of octaves to sum (fractal Brownian motion).

    Returns
    -------
    float or ndarray
        Noise value(s) between -1 and 1, depending on whether input is in scalar or array format.
    """
    if is_number(args[0]):
        scalar = True
        args = [np.ones(1)*a for a in args]
    else:
        scalar = False
    res = _value_funcs[len(args)-1](*args, octaves=octaves)
    if scalar:
        return res[0]
    return res

def grad_noise(*args, octaves=6):
    """
    Generate gradient noise (Perlin-like) in 1D, 2D, or 3D.

    Parameters
    ----------
    *args : float or ndarray
        Coordinates (x), (x, y), or (x, y, z). Each can be a scalar or numpy array.
    octaves : int, default=6
        Number of octaves to sum (fractal Brownian motion).

    Returns
    -------
    float or ndarray
        Noise value(s) between -1 and 1, depending on whether input is in
        scalar or array format.
    """
    if is_number(args[0]):
        scalar = True
        args = [np.ones(1)*a for a in args]
    else:
        scalar = False
    res = _grad_funcs[len(args)-1](*args, octaves=octaves)
    if scalar:
        return res[0]
    return res

def noise_grid(*args, gradient=True, octaves=6, mat=None, iter_mat=None):
    """
    Generate 2D grids or 3D slices of noise.

    Parameters
    ----------
    *args : ndarray
        For 2D: (x, y) coordinate arrays (1D each).
        For 3D slice: (x, y, z), where x, y are 1D arrays and z is either a scalar or an array.
    gradient : bool, default=True
        If True, generate gradient noise; if False, generate value noise.
    octaves : int, default=6
        Number of octaves to sum (fractal Brownian motion).

    Returns
    -------
    ndarray
        Noise values between -1 and 1 over the requested grid.
    """
    return _grid_funcs[len(args)-2][int(gradient)](*args, octaves=octaves, mat=mat, iter_mat=iter_mat)

# Internal state
state = lambda: None
state.sphere = None
state.circle = None
state.fractal_bounding = {}

def is_number(x):
    return isinstance(x, numbers.Number)

def splitmix64(x):
    # Adapted from https://nullprogram.com/blog/2018/07/31/
    x = np.array(x)
    x ^= x >> 30
    x *= 0xBF58476D1CE4E5B9
    x ^= x >> 27
    x *= 0x94D049BB133111EB
    x ^= x >> 31
    return x.astype(np.uint64)


# Murmur hashing adapted from https://www.shadertoy.com/view/ttc3zr
# Seems to be slightly faster and works well
M = np.uint64(0x5bd1e995)
H = np.uint64(1190494759)

def murmur11(x):
    x = x * M;
    x ^= x>>24; x *= M;
    h = H * M
    h ^= x;
    h ^= h>>13; h *= M; h ^= h>>15;
    return h

def fract(x):
    return x - np.floor(x) #np.modf(x)[1]

def uint64(x):
    x = np.asarray(np.floor(x)).astype(np.int64)
    x = x.view(np.uint64)
    return x

def to01(x):
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

# 2d and 3d hashes from Jarzynski and Olano (2019)
# https://www.jcgt.org/published/0009/03/02/
VA = np.uint64(1664525)
VB = np.uint64(1013904223)
s16 = np.uint64(16)

def pcg2d(v):
    v = v * VA + VB
    v[0] += v[1] * VA #1664525
    v[1] += v[0] * VA #1664525
    v = v ^ (v>>16)
    v[0] += v[1] * VA #1664525
    v[1] += v[0] * VA #1664525
    v = v ^ (v>>16)
    return v

def pcg3d(v):
    v = _u64(v)
    v = v * VA + VB
    v[0] += v[1]*v[2]
    v[1] += v[2]*v[0]
    v[2] += v[0]*v[1]
    v ^= v >> s16
    v[0] += v[1]*v[2]; v[1] += v[2]*v[0]; v[2] += v[0]*v[1];
    return v


hashf = murmur11 #
#hashf = splittable64
#hashf = splitmix64_mix # splitmix64_mix #fmix64
#hashf = fmix64 # Slow
#hashf = wyhash64

C1 = 0 #np.uint64(0x9E3779B97F4A7C15)
C2 = 0 #np.uint64(0xC2B2AE3D27D4EB4F)
C3 = 0 #np.uint64(0x165667B19E3779F9)

def _u64(a):
    return np.asarray(a, dtype=np.uint64)

# 1D combine (hash of x only)
def hash11(hx):
    hx = uint64(hx)
    C1 = np.uint64(0x9E3779B97F4A7C15)  # golden-ratio increment
    h  = np.uint64(cfg.seed)
    h  = hashf(h ^ (_u64(hx) + C1))
    return to01(h)*2 - 1

# 2D combine
def hash21i(hx, hy):
    hx, hy = [uint64(v) for v in [hx, hy]]
    h  = np.uint64(cfg.seed)
    h  = hashf(h ^ (_u64(hx) + C1))
    h  = hashf(h ^ (_u64(hy) + C2))
    return h

def hash21(hx, hy):
    return to01(hash21i(hx, hy))*2 - 1

# 3D combine
def hash31i(hx, hy, hz):
    hx, hy, hz = [uint64(v) for v in [hx, hy, hz]]
    h  = np.uint64(cfg.seed)
    h  = hashf(h ^ (_u64(hx) + C1))
    h  = hashf(h ^ (_u64(hy) + C2))
    h  = hashf(h ^ (_u64(hz) + C3))
    return h

def hash31(hx, hy, hz):
    return to01(hash31i(hx, hy, hz))*2 - 1

def hash33(hx, hy, hz):
    hx, hy, hz = [uint64(v + cfg.seed) for v in [hx, hy, hz]]
    return to01(pcg3d(np.stack([hx, hy, hz])))*2 - 1

def hash22(hx, hy):
    hx, hy = [uint64(v + cfg.seed) for v in [hx, hy]]
    return to01(np.stack([hx, hy]))*2 - 1

def mix(a, b, t):
    return a + (b - a)*t

def stack(v):
    return np.stack(v, axis=0)

def value_noise1(x):
    i = np.floor(x)
    f = fract(x)
    u = cfg.fade(f)
    return mix(hash11(i), hash11(i + 1), u)

def value_noise2(x, y):
    v = stack([x, y])
    i = np.floor(v)
    f = fract(v)

    a = hash21(i[0], i[1])
    b = hash21(i[0] + 1.0, i[1] + 0.0)
    c = hash21(i[0] + 0.0, i[1] + 1.0)
    d = hash21(i[0] + 1.0, i[1] + 1.0)

    # Same code, with the clamps in smoothstep and common subexpressions
    # optimized away.
    u = cfg.fade(f)
    return mix(a, b, u[0]) + (c - a) * u[1] * (1.0 - u[0]) + (d - b) * u[0] * u[1]

def value_noise3(x, y, z):
    v = stack([x, y, z])
    i = np.floor(v)
    ff = fract(v)
    u = cfg.fade(ff)

    a = hash31(v[0] + 0.0, v[1] + 0.0, v[2] + 0.0)
    b = hash31(v[0] + 1.0, v[1] + 0.0, v[2] + 0.0)
    c = hash31(v[0] + 0.0, v[1] + 1.0, v[2] + 0.0)
    d = hash31(v[0] + 1.0, v[1] + 1.0, v[2] + 0.0)
    e = hash31(v[0] + 0.0, v[1] + 0.0, v[2] + 1.0)
    f = hash31(v[0] + 1.0, v[1] + 0.0, v[2] + 1.0)
    g = hash31(v[0] + 0.0, v[1] + 1.0, v[2] + 1.0)
    h = hash31(v[0] + 1.0, v[1] + 1.0, v[2] + 1.0)

    return mix(mix(mix(a, b, u[0]),
                   mix(c, d, u[0]), u[1]),
               mix(mix(e, f, u[0]),
                   mix(g, h, u[0]), u[1]), u[2])

def grad_noise1(x):
    i = np.floor(x)
    f = fract(x)
    u = cfg.fade(f)
    g0 = hash11(i)
    g1 = hash11(i + 1)
    # From https://www.shadertoy.com/view/3sd3Rs
    return 2*mix( g0*(f-0.0), g1*(f-1.0), u)

def grad2(x, y):
    N = 64
    if state.sphere is None:
        t = np.linspace(0, np.pi*2, N, endpoint=False)
        state.circle = np.vstack([np.cos(t), np.sin(t)]).T
        np.random.default_rng(cfg.seed).shuffle(state.circle)
        state.circle = state.circle.T

    i = hash21i(x, y)&(N-1)
    return state.circle[:, i]

    # theta = hash21(x, y)*np.pi
    # return np.cos(theta), np.sin(theta)

def dotgrad21(x, y, ox, oy, f):
    gx, gy = grad2(x + ox, y + oy)
    return gx*(f[0]-ox) + gy*(f[1]-oy)

def grad_noise2(x, y):
    v = stack([x, y])
    i = np.floor(v)
    f = fract(v)
    u = cfg.fade(f)
    a = dotgrad21(i[0], i[1], 0.0, 0.0, f)
    b = dotgrad21(i[0], i[1], 1.0, 0.0, f)
    c = dotgrad21(i[0], i[1], 0.0, 1.0, f)
    d = dotgrad21(i[0], i[1], 1.0, 1.0, f)

    scale = 1.414213562373095 #1/sqrt(0.5**2 + 0.5**2)
    return mix(mix(a, b, u[0]),
               mix(c, d, u[0]), u[1])*scale

def fibonacci_sphere(samples=100):
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
    return np.array(points)

def grad3(x, y, z):
    N = 64
    if state.sphere is None:
        state.sphere = fibonacci_sphere(N)
        np.random.default_rng(cfg.seed).shuffle(state.sphere)
        state.sphere = state.sphere.T
    i = hash31i(x, y, z)&(N-1)
    return state.sphere[:, i]

    # theta, phi, _ = hash33(x, y, z)*np.pi
    # costh = np.cos(theta)
    # sinphi = np.sin(phi)
    # sinth = np.sin(theta)
    # return costh*sinphi, sinth*sinphi, costh

    # https://mathworld.wolfram.com/SpherePointPicking.html
    # Probably slow can do better
    # u, v, _ = hash33(x, y, z)*0.5 + 0.5
    # #v = hash31(x, y, z)*0.5 + 0.5
    # theta = 2*np.pi*u
    # phi = np.arccos(2*v - 1)
    # costh = np.cos(theta)
    # sinphi = np.sin(phi)
    # sinth = np.sin(theta)
    # return costh*sinphi, sinth*sinphi, costh


def dotgrad31(x, y, z, ox, oy, oz, f):
    gx, gy, gz = grad3(x + ox, y + oy, z + oz)
    return gx*(f[0]-ox) + gy*(f[1]-oy) + gz*(f[2]-oz)

def grad_noise3(x, y, z):
    v = stack([x, y, z])
    i = np.floor(v)
    ff = fract(v)
    u = cfg.fade(ff)

    a = dotgrad31(i[0], i[1], i[2], 0.0, 0.0, 0.0, ff)
    b = dotgrad31(i[0], i[1], i[2], 1.0, 0.0, 0.0, ff)
    c = dotgrad31(i[0], i[1], i[2], 0.0, 1.0, 0.0, ff)
    d = dotgrad31(i[0], i[1], i[2], 1.0, 1.0, 0.0, ff)
    e = dotgrad31(i[0], i[1], i[2], 0.0, 0.0, 1.0, ff)
    f = dotgrad31(i[0], i[1], i[2], 1.0, 0.0, 1.0, ff)
    g = dotgrad31(i[0], i[1], i[2], 0.0, 1.0, 1.0, ff)
    h = dotgrad31(i[0], i[1], i[2], 1.0, 1.0, 1.0, ff)

    scale = 1.1547005383792517 ##1/sqrt(0.5**2 + 0.5**2 + 0.5**3)
    return mix(mix(mix(a, b, u[0]),
                   mix(c, d, u[0]), u[1]),
               mix(mix(e, f, u[0]),
                   mix(g, h, u[0]), u[1]), u[2])*scale

def calc_fractal_bounding(octaves):
    '''Helper to scale the sum of octaves'''
    key = (octaves, cfg.falloff)
    if key in state.fractal_bounding:
        return state.fractal_bounding[key]
    v = 0.0
    a = 1.0
    amp = 0.0
    for i in range(octaves):
        amp += a
        a *= cfg.falloff
    bound = 1.0 / amp
    state.fractal_bounding[key] = bound
    return bound

    # r = cfg.falloff
    # if r == 1.0:
    #     fb = 1.0 / max(1, octaves)        # handle the r=1 edge case
    # else:
    #     fb = (1.0 - r) / (1.0 - r**octaves)

    # state.fractal_bounding[key] = fb
    # return fb

    # falloff = cfg.falloff
    # amp = falloff
    # amp_fractal = 1.0
    # for i in range(octaves):
    #     amp_fractal += amp
    #     amp *= falloff
    # fractal_bounding = 1 / amp_fractal
    # state.fractal_bounding[(octaves, cfg.falloff)] = fractal_bounding
    # return fractal_bounding

def make_fbm(func):
    def fbm(*args, octaves=8):
        v = 0.0
        a = calc_fractal_bounding(octaves)
        shift = cfg.shift
        x = np.stack(args, axis=0)
        for i in range(octaves):
            v += a * cfg.octave_map(func(*x))
            x = x * cfg.lacunarity + shift
            a *= cfg.falloff
        return v
    return fbm

value_fbm1 = make_fbm(value_noise1)
value_fbm2 = make_fbm(value_noise2)
value_fbm3 = make_fbm(value_noise3)
grad_fbm1 = make_fbm(grad_noise1)
grad_fbm2 = make_fbm(grad_noise2)
grad_fbm3 = make_fbm(grad_noise3)

def value_fbm_grid(x, y, octaves=8, mat=None, iter_mat=None):
    ''' Generate fractal value noise over a 2d grid defined by two 1d numpy arrays x, y'''
    v = 0.0
    a = calc_fractal_bounding(octaves)
    shift = cfg.shift
    xx, yy = np.meshgrid(x, y)
    if mat is not None:
        # for batch mul to work we need the dimension to be last col
        xx, yy = (np.stack([xx, yy], axis=-1)@mat.T).T
    for i in range(octaves):
        v += a * cfg.octave_map(value_noise2(xx, yy))
        xx = xx * cfg.lacunarity + shift
        yy = yy * cfg.lacunarity + shift
        if iter_mat is not None:
            xx, yy = (np.stack([xx, yy], axis=-1)@iter_mat.T).T
        a *= cfg.falloff
    return v

def grad_fbm_grid(x, y, octaves=8, mat=None, iter_mat=None):
    ''' Generate fractal gradient noise over a 2d grid defined by two 1d numpy arrays x, y'''
    v = 0.0
    a = calc_fractal_bounding(octaves)
    xx, yy = np.meshgrid(x, y)
    if mat is not None:
        # for batch mul to work we need the dimension to be last col
        xx, yy = (np.stack([xx, yy], axis=-1)@mat.T).T
    for i in range(octaves):
        v += a * cfg.octave_map(grad_noise2(xx, yy))
        xx = xx * cfg.lacunarity + cfg.shift
        yy = yy * cfg.lacunarity + cfg.shift
        if iter_mat is not None:
            xx, yy = (np.stack([xx, yy], axis=-1)@iter_mat.T).T
        a *= cfg.falloff
    return v

def value_fbm_grid3(x, y, z, octaves=8, mat=None, iter_mat=None):
    ''' Generate fractal value noise over a 2d grid as a slice of a 3d volume defined by two 1d numpy arrays x, y and a scalar z'''
    v = 0.0
    a = calc_fractal_bounding(octaves)
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)*z
    if mat is not None:
        # for batch mul to work we need the dimension to be last col
        xx, yy, zz = (np.stack([xx, yy, zz], axis=-1)@mat.T).T
    for i in range(octaves):
        v += a * cfg.octave_map(value_noise3(xx, yy, zz))
        xx = xx * cfg.lacunarity + cfg.shift
        yy = yy * cfg.lacunarity + cfg.shift
        if iter_mat is not None:
            xx, yy, zz = (np.stack([xx, yy, zz], axis=-1)@iter_mat.T).T
        a *= cfg.falloff
    return v


def grad_fbm_grid3(x, y, z, octaves=8, mat=None, iter_mat=None):
    ''' Generate fractal gradient noise over a 2d grid defined by two 1d numpy arrays x, y'''
    v = 0.0
    a = calc_fractal_bounding(octaves)
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)*z
    if mat is not None:
        # for batch mul to work we need the dimension to be last col
        xx, yy, zz = (np.stack([xx, yy, zz], axis=-1)@mat.T).T
    for i in range(octaves):
        v += a * cfg.octave_map(grad_noise3(xx, yy, zz))
        xx = xx * cfg.lacunarity + cfg.shift
        yy = yy * cfg.lacunarity + cfg.shift
        if iter_mat is not None:
            xx, yy, zz = (np.stack([xx, yy, zz], axis=-1)@iter_mat.T).T

        a *= cfg.falloff
    return v

_value_funcs = [value_fbm1,
               value_fbm2,
               value_fbm3]
_grad_funcs = [grad_fbm1,
               grad_fbm2,
               grad_fbm3]

_grid_funcs = [[value_fbm_grid,
                grad_fbm_grid],
               [value_fbm_grid3,
                grad_fbm_grid3]]
