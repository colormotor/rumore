#!/usr/bin/env python3
# from importlib import reload
# from . import rumore
# reload(rumore)
from .rumore import (cfg,
                    value_noise,
                    grad_noise,
                    noise_grid)

# Globals are handy but not great
# This resets everything when tate is persistent, e.g. in Python notebooks
def set_defaults():
    cfg.set_default()
