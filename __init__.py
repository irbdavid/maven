"""Routines to access and plot MAVEN L2 data.
"""

import celsius
from .maven_sc import *
import os
from glob import glob

__author__ = "David Andrews"
__copyright__ = "Copyright 2015, David Andrews"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "david.andrews@irfu.se"

DIRECTORY = os.getenv("SC_DATA_DIR", os.path.expanduser("~/data/")) + "maven/spg/data/misc/spice/naif/MAVEN/kernels/spk/"

# Update kernels if > 1wk old

# Load Kernels
load_kernels()

# Read orbits
orbits = celsius.OrbitDict()
for f in glob(DIRECTORY + 'maven_orb_rec_*.orb'):
    tmp = read_maven_orbits(f)

    for k, v in tmp.items():
        if k in orbits:
            raise IOError('Duplicate information contained in %s: Orbit %d repeated?' % (f, k))
        orbits[k] = v

if not orbits:
    raise IOError('No reconstructed orbits found?')

# Do these last:
tmp = read_maven_orbits(DIRECTORY + 'maven_orb_rec.orb')
for k, v in tmp.items():
    if k in orbits:
        raise IOError('Duplicate information contained in %s: Orbit %d repeated?' % (f, k))
    orbits[k] = v

print('Read information for %d orbits' % len(orbits))

stored_data = {}

# Instrument stuff
from . import lpw
from . import swea
from . import swia
from . import ngims
from . import kp
