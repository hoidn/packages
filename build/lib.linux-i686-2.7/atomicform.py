import numpy as np
import csv
import operator
import ipdb
import collections
from StringIO import StringIO
import re
import pkgutil

"""
compute the approximate atomic form factors of various atoms and ions 
using tabulated values of the fit coefficients
"""
#file with the tabulated coefficients
tableFile = 'data/all_atomic_ff_coeffs.txt'

#junk characters in the data file to get rid of
deleteChars = '\xe2\x80\x83'

hklList = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1], [3, 0, 0], [3, 1, 0], [3, 1, 1], [2, 2, 2]]

#positions of atoms of the two species
positions1 = ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))
positions2 = ((.5, .5, .5), (.5, 0, 0), (0, .5, 0), (0, 0, 0.5))


#bond locations for the unit cell based on the above positions. Note the 
#coordination number is 6, so there are 24 of these per unit cell
bondingLocations = [[0.5, 0.75, 0.5], [0.5, 0.25, 0.0], [0.0, 0.75, 0.0], [0.0, 0.25, 0.5], [0.5, 0.5, 0.75], [0.5, 0.0, 0.25], [0.0, 0.5, 0.25], [0.0, 0.0, 0.75], [0.75, 0.5, 0.5], [0.75, 0.0, 0.0], [0.25, 0.5, 0.0], [0.25, 0.0, 0.5], [0.25, 0.0, 0.0], [0.75, 0.5, 0.0], [0.75, 0.0, 0.5], [0.25, 0.5, 0.5], [0.0, 0.25, 0.0], [0.5, 0.75, 0.0], [0.5, 0.25, 0.5], [0.0, 0.75, 0.5], [0.0, 0.0, 0.25], [0.5, 0.5, 0.25], [0.5, 0.0, 0.75], [0.0, 0.5, 0.75]]

def cleanStr(chars, s):
    """delete all occurences of the characters in the string chars from the
       string s
    """
    r = re.compile(chars)
    return re.sub(r, '', s)


rawStr = cleanStr(deleteChars, pkgutil.get_data('atomicform', tableFile))
#rawStr = cleanStr(deleteChars, open(tableFile, 'r').read())

rawTable = np.genfromtxt(StringIO(rawStr), dtype=('S20', float, float, float, float, float, float, float, float, float), delimiter='\t')

#elementKeys = rawTable[rawTable.dtype.names[0]]
#numerical values of the table
zipped = np.array(zip(*rawTable))
elementKeys, values = zipped[0], zip(*zipped[1:])
values = np.array(values, dtype=float)

coeffDict = {k: v for (k, v) in zip(elementKeys, values)}

def getFofQ(k): 
    """
    return function that evaluates atomic form factor corresponding to the
    element/ion key, based on tabulated approximations.
    """
    if not coeffDict.has_key(k):
        raise ValueError, "valid keys are: " + str(coeffDict.keys())
    a1, b1, a2, b2, a3, b3, a4, b4, c = coeffDict[k]
    def FofQ(q):
        singleQ = lambda x :  a1 * np.exp(-b1 * (x/(4 * np.pi))**2)  +\
             a2 * np.exp(-b2 * (x/(4 * np.pi))**2)  + \
             a3 * np.exp(-b3 * (x/(4 * np.pi))**2)  + \
             a4 * np.exp(-b4 * (x/(4 * np.pi))**2) + c
        if isinstance(q, collections.Iterable): 
            return np.array(map(singleQ, q))
        else:
            return singleQ(q)
    return FofQ
    

def getPhase(rr, qq): 
    """evaluate exp(2 pi i q . r)"""
    return np.exp(2j * np.pi * np.dot(qq, rr)) 

def validCheck(qvec):
    """
    helper function to check and massage, if needed, an input array 
    of momentum transfers
    """
    errmsg  = "Momentum transfer must be represented by an array of length three"
    try: 
        if depth(qvec) == 1:
            qvec = [qvec]
        q11 = qvec[0][0]
    except:
        raise ValueError, errmsg
    else: 
        if  len(qvec[0]) != 3:
            raise ValueError,errmsg
    return qvec

def gaussianCloud(charge, x, y, z, sigma):
    """
    return function that evaluates the scattering amplitude of a 3d gaussian 
    charge distribution with standard deviation sigma, total charge charge, and
    centered at (x, y, z)
    """
    #the function that takes in a q vector and returns the amplitude
    tomap = lambda qq: charge * getPhase((x, y, z), qq) * \
            np.exp(- 0.5 * sigma**2 * np.dot(qq, qq))
    def amplitude(qvec):
        qvec = validCheck(qvec)
        return np.array(map(tomap, qvec))
    return amplitude

def fccStruct(a1, a2):
    """
    return function that evaluates the unit cell structure factor of 
    of an fcc material with two distinct species.
    The function expects the momentum transfer vector to be expressed in 
    terms of the reciprocal lattice basis vectors. 

    a1, a2: key strings for the elements
    """
    part1 = structFact(a1, positions1)
    part2 = structFact(a2, positions2)
    return lambda x: part1(x) + part2(x)

def structFact(species, positions):
    #form factors of the two species
    f = getFofQ(species)
    #function to evaluate amplitude contribution of a single atom
    def oneatom(formfactor, positions):
        return lambda qq: getPhase(positions, qq) * formfactor(map(np.linalg.norm, qq))
    #function to evaluate total amplitude for this strucure
    def amplitude(qvec):
        qvec = validCheck(qvec)
        return np.sum( np.array(map(lambda x: oneatom(f, x)(qvec), positions)), axis = 0) 
    return amplitude


def heat(qTransfer, structfact, donor = 'F', sigma = 0.05):
    """
    qTransfer: amount of charge to move
    structFac: structure factor functions. 

    the locations of the donor species are assumed to be positions2
    """
    perBond = float(qTransfer)/len(bondingLocations)
    gaussians = map(lambda x: gaussianCloud(perBond, x[0], x[1], x[2], sigma), bondingLocations)
    donors = structFact(donor, positions2)
    donorCharge = donors([0, 0, 0])
    scale = float(qTransfer)/donorCharge
    return  (lambda x: (structfact(x) - scale * donors(x) + np.sum(np.array([gaussian(x) for gaussian in gaussians]), axis = 0)))
    
    
    
def normHKLs(charges, alkali = 'Li', halide = 'F', hkls = hklList, mode = 'amplitude'):
    baseline = fccStruct(alkali, halide)
    if mode =='amplitude':
        mapFunc = lambda x: round(abs(x), 2)
    elif mode =='intensity':
        mapFunc = lambda x: round(abs(x)**2, 2)
    formFactors = np.array([map(mapFunc, heat(z, baseline)(hkls)) for z in charges])
    #normTable = lambda x: np.array(map(lambda y: abs(y), x))
    #return map(normTable, formFactors)
    return formFactors

def tableForm(charges, alkali = 'Li', halide = 'F', hkls = hklList, extracol = None, mode = 'amplitude'):
    f_hkls = normHKLs(charges, alkali = alkali, halide = halide, hkls = hkls, mode = mode)
    hklstrlist = hklString(hkls)
    if extracol != None:
        return np.array(zip(*np.vstack((hklstrlist, f_hkls, extracol))))
    else: 
        return np.array(zip(*np.vstack((hklstrlist, f_hkls))))

#def hklPermutations(hmax):
#    ipdb.set_trace()
#    return _hklPermutations(0, 0, 0, hmax, [[0, 0, 0]])

#def _hklPermutations(h, k, l, hmax, acc):
#    if l < k:
#        l += 1
#        return _hklPermutations(h, k, l, hmax, acc + [[h, k, l]])
#    elif k < h: 
#        k += 1
#        return _hklPermutations(h, k, l, hmax, acc + [[h, k, l]])
#    elif h <= hmax: 
#        h += 1
#        return  _hklPermutations(h, k, l, hmax, acc + [[h, k, l]])
#    else:
#        return acc
#
        

def FCChkl(maxh, complement = False):
    """allowed fcc reflections, up to maxh maxh maxh"""
    outlist = []
    for i in range(maxh + 1):
        for j in range(i + 1):
            for k in range(j + 1):
                #allowed reflections: h, k, l all even or all odd
                if not complement: 
                    if (i%2 == 0 and j%2 == 0 and k%2 ==0 ) or \
                            (i%2 == 1 and j%2 == 1 and k%2 ==1 ):
                        outlist += [[i, j, k]]
                else: 
                    if not ((i%2 == 0 and j%2 == 0 and k%2 ==0 ) or \
                            (i%2 == 1 and j%2 == 1 and k%2 ==1 )):
                        outlist += [[i, j, k]]
    return outlist

def hklString(hkl):
    """string representation of list of three integers"""
    def stringify(x):
        hklstr = map(str, x)
        return hklstr[0] + ';' + hklstr[1]  + ';' + hklstr[2]
    try: 
        hklStringList = map(stringify, hkl)
    except: 
        return stringify(hkl)
    else:
        return hklStringList 

def sortHKL(hkllist):
    """sort a list of hkls by momentum transfer"""
    def qTrans(hkl):
        return sum(map(lambda x: x**2, hkl))
    return sorted(hkllist, key = qTrans)

def depth(l):
    """evaluate maximum depth of a list or numpy array"""
    if isinstance(l, (np.ndarray, list)):
        return 1 + max(depth(item) for item in l)
    else:
        return 0

def csvWrite(fname, arr):
    with open(fname, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(arr)
        fp.close()
