#TODO: understand the weirdness with different behavior of += in different
#situations, specifically, why doesn't it work for incrementing the value of a
#data member?

#TODO: fix the energy out of range errors (seems to do it above 10 keV)

import ipdb
import matplotlib.pyplot as plt
import pdb
import random
import numpy as np
import mu
from scipy.interpolate import interp1d

class Photon: 
    """Stores position and direction, and energy of a  particle
       fields:
       energy: float or int, units eV
       position: float or int (i.e., depth in sample). Defaults to [0,0,0] 
       escapeState: 'below' if photon escaped below sample z > thickness), 
           'above' if photon escaped above sample, None otherwise
       direction: 3D unit vector. Defaults to None upon initialization if no direction vector is given

       It is assumed that the client will not revert direction to None
       after initialization
    """
    def __init__(self, energy, position = [0, 0, 0], direction = None, \
        escapeState = None):
        self.escapeState = escapeState
        self.energy = energy() 
        self.position = np.array(position)
#        if not direction and not self.direction: 
#            direction = randVec(1)[0]
        if direction != None: 
            assert np.abs(np.dot(direction, direction) - 1.) < 0.01, \
                "direction vector not normalized or invalid format"
        self.direction = direction

    def changeDirection(self, direction = None): 
        """Change Photon's direction (default to random if no argument given)"""
        if not direction: 
            direction = randVec(1)[0]
        self.direction = direction


#pre: fractions must sum to 1
class AlloySim:
    """
    arguments: 
    elementList: list of string abbreviation of elements
    eMax, eMin: min and max energies for which to calculate output photons
    thickness: thickness in cm of the sample. Note that the sample is 
        taken to extend from z = 0 to z = thickness
    eProbe: energy in eV of incident photons

    fields: 
    fractionsDict: k, v: element abbreviation, fraction in alloy
    dataDict: k, v: element abbreviation, corresponding ElementData instance
    thickness: equivalent to input args of thos names
    muTot: total mu of the alloy (sum weighted by element fractions)
    TODO: describe self.branchInfo
    """
    #number of points in energy grid
    GRIDPOINTS = 1000
    def __init__(self, elementList, fractions, eMin = 1000, eMax = 22000, thickness = .1):
        self.elementList = elementList
        self.eMin = eMin
        self.eMax = eMax
#        pdb.set_trace()
        elementDatas = [mu.ElementData(element, eMin = int(eMin * 0.8), eMax = int(eMax * 1.2)) for element in elementList]
        #properties of the constituent elements
        self.fractionsDict = dict((k, v) for k, v in zip(elementList, fractions))
        self.dataDict = dict((k, v) for k, v in zip(elementList, elementDatas))
        #properties of the sample and experimental configuration
        #sample thickness in cm
        self.thickness = thickness
        #density of the alloy
        self.cumulativeDensity = [np.array(self.fractionsDict.values()) * np.array([elem.density for elem in self.dataDict.values()])][0]
        eGrid = np.linspace(eMin, eMax, AlloySim.GRIDPOINTS)
        muTot = np.linspace(0, 0, AlloySim.GRIDPOINTS)
        
        #total mu of alloy sample
        for element in elementList: 
            #pdb.set_trace()
            muTot = muTot + self.dataDict[element].mu(eGrid) * self.fractionsDict[element]

        self.muTot = interp1d(eGrid, muTot)

    def sampLength(self, energy, num = 1): 
        meanDensity = list(accumu(self.cumulativeDensity))[-1]
        #total attenuation factor of the alloy at this energy
        #totalMu = float(self.muTot(energy)) * meanDensity
        totalMu = float(self.muTot(energy)) 
        #TODO: use size kwarg for np.random.exponential
        interactionLengths = [np.random.exponential(1/totalMu, 1)[0] for i in xrange(num)]
        return interactionLengths


    def sampEnergies(self, energy): 
        ipdb.set_trace()
        assert energy > self.eMin and energy < self.eMax, "energy out of range"
        alloys = self.dataDict.values()

        #mu in units 1/cm for all alloy components, multiplied by the composition fraction of each acomponent
        weights = (np.array([float(alloy.muCached(energy)) for  alloy in alloys]) * np.array(self.cumulativeDensity))
        #normalize weights to 1
        weightsTot = list(accumu(weights))[-1]
        weights = weights/weightsTot

        #probability for an abosrption in each element resulting in core hole
        holeProbs = [float(alloy.mu1sCached(energy))/float(alloy.muCached(energy)) for alloy in alloys]

        #probability for a core hole to produce a fluorescence photon
        Kbranches = [alloy.kBranch for alloy in alloys]

        #probabilites for fluorescence photon of each element to be Ka2, Ka1, or Kb
        photonBranches = getABratio(alloys)
        #photBranchesTot = [float(list(accumu(element))[-1]) for element in photonBranches]
        photBranchesTot = [float(np.sum(element)) for element in photonBranches]
        #normalized to sum to 1
        photonBranches = [np.array(photonBranches[i])/photBranchesTot[i] for i in xrange(len(photonBranches))]

        #energies of Ka2, Ka1, and Kb for each alloy component
        energyChoices = self.getABenergies(sort = False)
        #flatten energyChoices 
        photonEnergies = []
        for element in energyChoices: 
            photonEnergies += list(element)
        photonEnergies = [None] + photonEnergies

        #overall probability for the incident photon to produce photons of each energy in photonEnergies
        photonProbabilities = np.einsum('i,ij->ij', np.array(weights) * np.array(holeProbs) * np.array(Kbranches), np.array(photonBranches))
        #flatten photonProbabilities
        newPhotonProbabilities = []
        for element in photonProbabilities: 
            newPhotonProbabilities += list(element)
        photonProbabilities = [1 - sum(newPhotonProbabilities)] + \
             newPhotonProbabilities

        return [photonEnergies, photonProbabilities]

    def getABenergies(self, mode = 'normal', sort = True):
        """ return list of energies for  Ka2, Ka1, and 
            Kb emission 
    
            elDatList is a list of ElementData objects
    
            mode = 'normal': return list of depth 2. 
            mode = 'flat': return flattened list

            sort: return values sorted in order corresponding to increasing
            atomic number. 
        """
        #pdb.set_trace()
        keylist = self.dataDict.keys()
        if sort: 
            keylist.sort()
        values = [self.dataDict[k] for k in keylist]
        energies = [map(lambda x: x.energy, val.fDict.values()) for val in values]
        if mode == 'flat': 
            newEnergies = []
            for elem in energies: 
                newEnergies += elem
            return newEnergies
        return energies
    

    def stepPhoton(self, photon, distance): 
        """Propagate the Photon instance photon through the alloy slab

           This function updates Photon.escapeState. it modifies position but 
           NOT direction. 
           It takes responsibility to not modify the position of an absorbed
           or escaped photon. 
        """
        #photon has been neither absorbed nor has it escaped
        if photon.direction != None and not photon.escapeState:
            photon.position = photon.position + photon.direction * distance 
        z = photon.position[2]
        if z > 0: 
            photon.escapeState = 'above'
        elif z < -self.thickness: 
            photon.escapeState = 'below'

    def runSim(self, eIncident, nPhotons = 1):  
        """Initialize the simulation by creating a list of downward-traveling 
           Photons with energy eIncident. Then call a recursive helper 
           function that evolves individual photons. 

           eIncident may be a float, int, or function  of no arguments 
           that returns one of these.

           Return a list of Photon instances
        """
        photons = []
        #make eIncident callable if it isn't 
        if not hasattr(eIncident, '__call__'):
            eIncident = lambda: eIncident
        for i in xrange(nPhotons):
            newPhoton = Photon(eIncident, direction = np.array([0, 0, -1]))
            photons += [newPhoton]
        
        for phot in photons: 
            self.evolvePhoton(phot)

        return photons

    def evolvePhoton(self, photon): 
        """ Recursive function to evolve a single photon. 
        
            direction = None denotes an absorbed photon. 
        """
        #TODO: since only a few photon energies appear in the simulation, 
        #calling sampEnergies every time this function is invokes is a waste.
        #pdb.set_trace()
        if photon.energy and not photon.escapeState:
            [fEnergies, fProbs]  = self.sampEnergies(photon.energy)
            distance = self.sampLength(photon.energy, 1)[0]
            self.stepPhoton(photon, distance) 
            #p.photon = energy
            if not photon.escapeState: 
                newEnergy = fEnergies[weighted_choice(fProbs)] 
                photon.energy = newEnergy
                #pick random direction
                if newEnergy: 
                    photon.changeDirection()
                    #recurse only if fluorescence photon was produced
                    self.evolvePhoton(photon)
                else: 
                    photon.direction = None

def getABratio(elDatList):
    """ return list of branching probabilities to Ka2, Ka1, and 
        Kb. 

        elDatList is a list of ElementData objects
    """
    probs = [map(lambda x: x.intensity, eldat.fDict.values()) for eldat in elDatList]
    return probs

def getABenergies(elDatList, mode = 'normal'):
    """ return list of energies for  Ka2, Ka1, and 
        Kb emission 

        elDatList is a list of ElementData objects

        mode = 'normal': return list of depth 2. 
        mode = 'flat': return flattened list
    """
    energies = [map(lambda x: x.energy, eldat.fDict.values()) for eldat in elDatList]
    if mode == 'flat': 
        newEnergies = []
        for elem in energies: 
            newEnergies += elem
        return newEnergies
    return energies

#given a list of weights, return an index with probability proportional
#to the value at that index. If values are provided it returns the selected
#value, not an index
def weighted_choice(weights, values = None):
    if values: 
        assert len(weights) == len(values), "mismatch between length of weight and value lists"
    totals = []
    running_total = 0
    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            if values: 
                return values[i]
            else:
                return i
	
def accumu(lis):
    """return an iterator that computes the accumulated sum of lis"""
    summ=0
    for x in lis:
        summ+=x
        yield summ


def randVec(num = 1): 
    """return list of cartesian coords for randomly-selected 3D unit vectors"""
    flat = np.random.normal(size= 3 * num)
    np.random.shuffle(flat)
    vectors = np.reshape(flat, (num, 3))
    result = np.array([vec/np.sqrt(np.dot(vec, vec)) for vec in vectors])
    return result


def histoSim(elementList, fractions, energy, thickness, num = 1, bins = 60): 
    """ Use the AlloySim class to perform an alloy fluorescence simulation. 

        thickness units are cm. 

        plot a histogram of number of counts vs energy
    """
    energies = getSimDat(elementList, fractions, energy, thickness, num, bins = 100)
    plt.hist(energies, bins = bins)
    plt.show()

def getSimDat(energy, elementList = None, fractions = None, thickness = None, num = 1, sim = None):
    """run fluorescence simulation and return list of fnal photons in the 
       format [energies, counts]. 
       

       helper for histoSim. 

       All other aruments are ignored if sim is provided
    """
    if not sim: 
        assert all([elementList, fractions, thickness]), "elementList, fractions,\            and thickness must be provided."
        sim = AlloySim(elementList, fractions, thickness = thickness)
    energies = simDatRaw(energy, sim=sim, direction=False, num=num)['data']
    #list of unique energies
    return tallyUnique(energy, energies, sim)

def tallyUnique(energy, energies, sim): 
    """return the list of counts per each output energy given an 
       AlloySim object and a list of photon output energies
       from a simulation, and the energy at which the simulation
       was performed
    """
    energies = list(energies)
    uniqE =  sim.getABenergies(mode = 'flat') + [0, energy]
    #make a tally
    counts = map(energies.count, uniqE)
    return [uniqE, counts]
   

def simDatRaw(energy, elementList = None, fractions = None, thickness = None, num = 1, sim = None, direction=False):
    """run fluorescence simulation. If direction == False return 1D list 
       of photon energies. If direction == True return 
       [energies, directions]
       

       All other aruments are ignored if sim is provided
    """
    if not sim: 
        assert all([elementList, fractions, thickness]), "elementList, fractions,\            and thickness must be provided."
        sim = AlloySim(elementList, fractions, thickness = thickness)
    photons = sim.runSim(energy, num)
    energies = [photons[i].energy for i in range(num)]
    #replace None with 0 to denote absorbed photons
    if direction == False: 
        energies = [i if i != None else 0 for i in energies]
        return {'sim':sim, 'data': energies}
    else: 
        directions = [photons[i].direction for i in range(num)]
        return {'sim': sim, 'data': [np.array(energies), np.array(directions)]}
    


