#look up k edge energies from edges.dat
import os
import numpy as np


def edge_energy(atomicN):
    """return energy in units of eV"""
    if os.name == 'nt':
        path = 'E:\\Dropbox\\Seidler_Lab\\physical_data\\edges.dat'
    else:
        path = '/home/oliver/Dropbox/Seidler_Lab/physical_data/edges.dat'
    if (not type(atomicN) == int) or (atomicN < 4):
        raise ValueError('invalid atomic number: ' + str(atomicN))
    tab = np.genfromtxt(path)
    return 1000 * tab[atomicN - 4]
        
