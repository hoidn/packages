import matplotlib.pyplot as plt
import pdb
import random
import numpy as np
import mu
from scipy.interpolate import interp1d
import scipy.optimize as opt
import alloySim
import itertools
import copyreg
import types

import multiprocessing
#from multiprocessing import Process, Pipe


#TODO: the simulation chokes when incident energy is below that of the K edge in one or more of the constituent species
#temporary fix: increase incident energy to 10 keV

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in zip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]

def flatten2D(ls): 
    """flatten a 2D list (NOT numpy array)"""
    outlist = []
    for l in ls: 
        outlist += l
    return outlist

def makeCombos(elements = [22,24,26,28,30], comboSizes = [2,3]):
    iters  = [itertools.combinations(elements, size) for size in comboSizes]
    comboLists = [[list(np.sort(combo)) for combo in it] for it in iters]
    return flatten2D(comboLists)

#def testFit(): 
#    elements = [22,24,26]
#    f0 = [0.33, 0.33, 0.34]
#    f = opt.basinhopping(scoreAlloy, f0, stepsize = 0.1, niter = 10)
#    return f

#def simAll(elements = [22,24,26,28,30], comboSizes = [2,3]):
#    #pdb.set_trace()
#    #pool = multiprocessing.Pool(5)
#    combinations = makeCombos(elements, comboSizes)
#    optima = []
#    allStartFracs = [[1./len(combo)] * len(combo) for combo in combinations]
#    allArgs = zip(allStartFracs, combinations)
#    
#    optima = parmap(lambda x: iterScore(x[0], x[1]), allArgs)
#
#    return zip(combinations, optima)

def simAll(elements = [22,24,26,28,30], comboSizes = [2,3], fractDict = None, mode ='run', num = 100000, angleRange=(np.deg2rad(0), np.deg2rad(90)), energy = 10000, thickness = 0.0005):
    """Fractdict contains element list: fractions pairs. If Fractdict is 
       supplied, elements argument is ignored.
    """ 
    #pdb.set_trace()
    pool = multiprocessing.Pool(8)
    optima = []
    if mode == 'run':
        if fractDict == None: 
            combinations = makeCombos(elements, comboSizes)
            allStartFracs = [[1./len(combo)] * len(combo) for combo in combinations]
        else: 
            allStartFracs, combinations = list(fractDict.values()), [list(map(int, x)) for x in list(fractDict.keys())]
            #histo = [binSim(elements, fracs, thickness = 0.0005, energy = 10000, num = num) for elements, fracs in zip(combinations, allStartFracs)]
            histo = [runAndTally(energy=energy, elements = elements, fractions = fracs, thickness = thickness, num = num, angmin = angleRange[0], angmax = angleRange[1]) for elements, fracs in zip(combinations, allStartFracs)]
        return histo
    elif mode =='optimize':
        allStartFracs, combinations = list(fractDict.values()), [list(map(int, x)) for x in list(fractDict.keys())]
        allArgs = list(zip(list(map(list, allStartFracs)), list(map(list, combinations))))
        optima = pool.map(iterScoreTop, allArgs)
        return list(zip(combinations, optima))
    else: 
        print("mode " + mode + "unrecognized")

#TODO: find sensoble way of passing angle range argument
def iterScoreTop(arguments, angleRange=(np.deg2rad(0), np.deg2rad(90)), target = None):
    result =  iterScore(arguments[0], arguments[1], angleRange=angleRange, \
            target = target)
    print(arguments[1])
    print(result)
    return result
   
#    for combo in combinations: 
#        startFracs = [1./len(combo) for i in range(len(combo))]
#        optima += [iterScore(startFracs, elements = combo, increment = 0.02, nIters = 90)]
#    return zip(combinations, optima)

def iterScore(startFractions, elements = [22,24,26], thickness = 0.0005, energy = 10000, num = 10000, increment = .01, nIters = 45, angleRange=(np.deg2rad(0), np.deg2rad(90)), target = None):
    #pdb.set_trace()
    newFractions = startFractions

    pdb.set_trace()
    histo = runAndTally(energy, elements = elements, fractions = startFractions, thickness = thickness, num = num, angmin = angleRange[0], angmax = angleRange[1])
    #portion of the histogram  with fluorescence lines only
    #fLines = np.array(histo)[:,:len(histo[0]) - 2]
    kAlphas = np.array([histo[1][2*i] for i in range(len(elements))], dtype = np.dtype('float'))
    if target != None: 
        try: 
            target = np.array(target)
        except:
            raise ValueError("target must be sequence with same shape as elements")
        else:
            kAlphas = kAlphas/target
    kaTot = np.sum(kAlphas)
    kaMean = kaTot/len(elements)
    diffs = np.abs(kAlphas - kaMean)

    worst = max(diffs)
    worstIndx = list(diffs).index(worst)

    if kAlphas[worstIndx] - kaMean > 0: 
        correction = -increment
    else: 
        correction = increment
    
    for i in range(len(elements)): 
        if i == worstIndx: 
            newFractions[i] += correction
        else: 
            newFractions[i] -= correction/(len(elements) - 1)

    nIters -= 1
    if nIters  < 1:
        return newFractions
    else: 
        return iterScore(newFractions, elements, thickness, energy, num, increment, nIters, target = target)

    

    badness = np.dot(diffs, diffs)
    return badness
    
def runAndTally(energy, elements = [22, 24, 26], fractions = [0.33, 0.33, 0.34], thickness = 0.0005, num = 10000, angmin= np.deg2rad(0), angmax = np.deg2rad(90)):
    #this is a dict with keys 'sim' and 'data'
    unfiltered = alloySim.simDatRaw(energy, elementList=elements, fractions=fractions, thickness=thickness, num=num, direction=True)
    
    filtered = filterAngles(*unfiltered['data'], angmin = angmin, angmax = angmax)
    
    tallied =  alloySim.tallyUnique(energy, filtered, unfiltered['sim'])
    
    tallyABcombined = combineAB(tallied[0], tallied[1], unfiltered['sim'])
    return tallyABcombined

#def scoreAlloy(fractions, elements= [22,24,26], thickness = 0.005, energy = 8000, num = 10000):
#    """
#       Return a quality criterion for the given simulation parameters. 
#
#       The desired result is equal magnitudes for all the K alpha peaks. 
#    """
#    histo = binSim(elements, fractions, thickness, energy, num)
#    #portion of the histogram  with fluorescence lines only
#    #fLines = np.array(histo)[:,:len(histo[0]) - 2]
#    kAlphas = np.array([histo[1][2*i] for i in range(len(elements))], dtype = np.dtype('float'))
#    kaTot = np.sum(kAlphas)
#    kaMean = kaTot/len(elements)
#    diffs = kAlphas - kaMean
#    badness = np.dot(diffs, diffs)
#    return badness
#    
#
##TODO: docstring
#def binSim(elements, fractions, thickness, energy, num):
#    """ with the current structure of alloySim and the input data, k alpha 1
#        and 2 are considered separately. 
#    """
#    sim  = alloySim.AlloySim(elements, fractions, thickness = thickness) 
#    energies = sim.getABenergies()
#
#    #assume that the ordering of flatEnergies is the same as that returned
#    #by alloySim.getABenergies(mode = 'flat'). Also assuming that each element
#    #has three associated fluorescence lines. The two additional energies (0 
#    #and energy) are added to the end
#    [flatEnergies, counts] = alloySim.getSimDat(energy, sim = sim, num = num)
#
#
#    countsNew = []
#    energiesNew = []
#    for i in range(len(energies)):
#        energiesNew += [np.mean(flatEnergies[3*i:3*i + 2]), flatEnergies[3*i + 2]]
#        countsNew += [np.sum(counts[3*i:3*i + 2]), counts[3*i + 2]]
#
#    energiesNew += flatEnergies[-2:]
#    countsNew += counts[-2:]
#    
#    return [energiesNew, countsNew]

def combineAB(allenergies, allcounts, sim):
    """ with the current structure of alloySim and the input data, k alpha 1
        and 2 are considered separately. 
    """
    energies = sim.getABenergies()

    #assume that the ordering of flatEnergies is the same as that returned
    #by alloySim.getABenergies(mode = 'flat'). Also assuming that each element
    #has three associated fluorescence lines. The two additional energies (0 
    #and energy) are added to the end
    allcountsNew = []
    energiesNew = []
    for i in range(len(energies)):
        energiesNew += [np.mean(allenergies[3*i:3*i + 2]), allenergies[3*i + 2]]
        allcountsNew += [np.sum(allcounts[3*i:3*i + 2]), allcounts[3*i + 2]]
    energiesNew += allenergies[-2:]
    allcountsNew += allcounts[-2:]
    return [energiesNew, allcountsNew]
    
    
def filterAngles(energies, directions, angmin=np.deg2rad(0), angmax=np.deg2rad(90)): 
    """filter the output of alloySim.simDatRaw to include only photons 
       with final directions equivalent to scattering angles between 
       angmin and angmax (in radians)

       energies and directions must be nump.ndarray
    """
    energies = np.array([_f for _f in energies if _f])
    directions = np.array([x for x in directions if type(x) != type(None)])
    directions = spherical(directions)
    return energies[np.logical_and(angmax > directions[:,1], directions[:,1] > angmin)]

#def isBetween(

def spherical(xyz):
    """return equivalent spherical coords for the given list of 3d 
       cartesian coords in format [r, theta, phi]"""
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew


    

def approx_equal(a, b, tol = 0.0001):
    """equality test for floats"""
    return abs(a - b) < tol

def getKabRatio(simDat):
    """return kalpha/kbeta ratio from simDat (format is output format of binSim)
    """
    abList = []
    counts = simDat[1][:-2]
    for i in range(len(counts)/2): 
        abList += [counts[2*i]/float(counts[2*i + 1])]
    return abList
