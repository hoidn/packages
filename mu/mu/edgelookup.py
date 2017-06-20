#look up k edge energies from edges.dat
import os
import numpy as np
from utils import utils

PKG_NAME = __name__.split('.')[0]

def edge_energy(atomicN):
    """return energy in units of eV"""
    path =  utils.resource_path('data/edges.dat', pkg_name = PKG_NAME)
    if (not type(atomicN) == int) or (atomicN < 4):
        raise ValueError('invalid atomic number: ' + str(atomicN))
    tab = np.genfromtxt(path)
    return 1000 * tab[atomicN - 4]
        
