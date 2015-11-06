import re
import os
from StringIO import StringIO
import numpy as np
from scipy.interpolate import interp1d
import urllib

import edgelookup

#class to store info on single fluorescence line
class Fluorescence: 
    """
        fields: 
        name: 
        atomicNumber: 
        intensity: relative intensity of fluorescence line.
        units: units of intensity
    """
    def __init__(self, atomicNumber = None, name = None, energy = None, intensity = None, units = None, initList = None):
        """
        arguments: 
        atomicNumber, name, energy, intensity correspond to same-named fields
        intensityList: list containing the above information in the output 
            format of alphaBetaBranch (energy, atomicNumber, name, intensity)
        """
        self.name = name
        self.atomicNumber =  atomicNumber
        self.energy = energy
        self.intensity = intensity
        self.units = units 
        #format: (float, int,  np.dtype('S4'), float))
        if initList: 
            self.energy, self.atomicNumber, self.name, self.intensity = list(initList)

#class to store data on an element
class ElementData:
    """
        atomicN: atomic number
        density: ambient density of the element (downloaded from mathematica) in
        units g/cm**3
        fDict: dictionary of fluorescence lines (keys are line name)
        kBranch: probability of 1s core hole to produce fluorescence
        nu, mu1s: attenuation factors in units cm^2/g
        
    """
    def __init__(self, atomicN, eMin = 1000, eMax = 12000): 
        """ 
            eMin and eMax: energies (in keV) for which mu and other energy-
                dependent quantities will be calculated. 
        """
        assert isinstance(atomicN, int),  "atomicN is not a positive integer "
        self.N = atomicN
        self.name = getElementName(atomicN)
        self.density = getElementDensity(self.name)[2]
        self.mu = getMu(self.name, eMin, eMax, interp = True)
        #dict with previously-evaluated values of mu
        self.mu_cache = {}
        self.mu1s = get1s(self.name, eMin, eMax, self.N, interp = True)
        #dict with previously-evaluated values of mu1s
        self.mu1s_cache = {}
        self.kBranch = Kbranch(self.name)["ratio"]
        initList = alphaBetaBranch(self.name)
        fList = [Fluorescence(initList = line) for line in initList] 
        keys = [ls[2] for ls in initList]
        self.fDict = {key: value for (key, value) in zip(keys, fList)}
#        self.fDict = {fluo[2]: [fluo[0], fluo[1], fluo[3]]}
#        self.fDict = fList

    def mu1sCached(self, energy):
        try: 
            val = self.mu1s_cache[energy]
        except: 
            val = self.mu1s(energy)
            self.mu1s_cache[energy] = val
        return val


    def muCached(self, energy):
        try: 
            val = self.mu_cache[energy]
        except: 
            val = self.mu(energy)
            self.mu_cache[energy] = val
        return val



def getMu(element, eMin, eMax, interp = True): 
    """Get mu(E), (units eV for E, 1/cm for mu) for element from FFAST
       database. Returns an interpolation function by default. 
    """
    energy, f1, f2, photo, allscatter, total, kphoto, lambd = np.array(zip(*getFFAST(element, eMin, eMax)))
    energy = 1000. * energy
    if interp: 
        return interp1d(energy, total)
    else: 
        return [energy, total]

def getFFAST(element, eMin, eMax): 
    """download data for given element from the FFAST database"""
    assert eMax >= 1000, "units are eV. eMax must be >= 1000."
    #template URL for FFAST data
    template = "http://physics.nist.gov/cgi-bin/ffast/ffast.pl?&Formula=<name>\
&gtype=4&range=S&lower=<emin>&upper=<emax>"
    [eMin, eMax] = map(lambda x: str(int(x/1000)), [eMin, eMax])
    url = replaceAll({'<name>': element, '<emin>': eMin, '<emax>': eMax}, template)
    f = urllib.urlopen(url)
    s = f.read()
    f.close()
    datpat = re.compile('.*nm\\n(.*?)\\n<.*$', re.DOTALL)
    datsearch = re.search(datpat, s)
    datstring = datsearch.group(1)
    datls = np.genfromtxt(StringIO(datstring), dtype = float)
    return datls

def replaceAll(rep, text): 
    """
    substitute regexes in the text according to the rules given by the
    dictionary rep. 
    """
    rep = dict((re.escape(k), v) for k, v in rep.iteritems())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    return text

#class to store data on an element
class ElementData:
    """
        atomicN: atomic number
        density: ambient density of the element (downloaded from mathematica) in
        units g/cm**3
        fDict: dictionary of fluorescence lines (keys are line name)
        kBranch: probability of 1s core hole to produce fluorescence
        nu, mu1s: attenuation factors in units cm^2/g
        
    """
    def __init__(self, atomicN, eMin = 1000, eMax = 12000): 
        """ 
            eMin and eMax: energies (in keV) for which mu and other energy-
                dependent quantities will be calculated. 
        """
        assert isinstance(atomicN, int),  "atomicN is not a positive integer "
        self.N = atomicN
        self.name = getElementName(atomicN)
        self.density = getElementDensity(self.name)[2]
        self.mu = getMu(self.name, eMin, eMax, interp = True)
        #dict with previously-evaluated values of mu
        self.mu_cache = {}
        self.mu1s = get1s(self.name, eMin, eMax, self.N, interp = True)
        #dict with previously-evaluated values of mu1s
        self.mu1s_cache = {}
        #self.kBranch = Kbranch(self.name)["ratio"]
        initList = alphaBetaBranch(self.name)
        fList = [Fluorescence(initList = line) for line in initList] 
        keys = [ls[2] for ls in initList]
        self.fDict = {key: value for (key, value) in zip(keys, fList)}
#        self.fDict = {fluo[2]: [fluo[0], fluo[1], fluo[3]]}
#        self.fDict = fList

    def mu1sCached(self, energy):
        try: 
            val = self.mu1s_cache[energy]
        except: 
            val = self.mu1s(energy)
            self.mu1s_cache[energy] = val
        return val


    def muCached(self, energy):
        try: 
            val = self.mu_cache[energy]
        except: 
            val = self.mu(energy)
            self.mu_cache[energy] = val
        return val

#given element name, eMin and eMax in eV, locate the K-edge and subtract
# out pre-edge absorption with a 1/E^3 dependence, and return resulting mu(E) 
#in same format as getFFAST
def get1s(element, eMin, eMax, atomicN, interp = False):
    muofE = np.array(getMu(element, eMin, eMax, interp = False))
    #pdb.set_trace()
    #KEnergy = Kbranch(element)["energy"]
    KEnergy = edgelookup.edge_energy(atomicN)
    energies = muofE[0]
    diffs = energies - KEnergy
    diffs = map(lambda x: x if x > 0 else np.inf, diffs)
    edgeIndx = diffs.index(min(diffs))
    [EPreEdge, muPreEdge] = [muofE[0][edgeIndx - 1], muofE[1][edgeIndx - 1]]
    
    subMu = muPreEdge * (EPreEdge**3)/((energies)**3)
    mu1s = zip(energies, muofE[1] - subMu)
    mu1s = [0 if pair[0] < EPreEdge else pair[1] for pair in mu1s]

    if interp: 
        return interp1d(energies, mu1s)
    else: 
        return [energies, mu1s]
    

def getAllFluorescence(): 
    edgedatf = open(edgedatdir, 'r')
    edgedat = edgedatf.read()
    edgedat = replaceAll({'KL3':'ka1', 'KM3':'kb1'}, edgedat)
    edgetbl = np.genfromtxt(StringIO(edgedat), delimiter='\t', dtype = (np.dtype('S100'), float, np.dtype('S100'), float, float, float, float, float, np.dtype('S100')))
    element, junk1, line, theoryE, junk2, experimentE, junk5, junk6, shell = zip(*edgetbl)
    return [element, experimentE, line]

def getElementFluorescence(element): 
    fluoDat = getAllFluorescence()
    return filter(lambda x : x[0] == element, zip(*fluoDat))

def getElementDensities(): 
    if os.name == 'nt':
        elementDatDir = 'E:\\Dropbox\\Seidler_Lab\\physical_data\\elementDensities.csv'
    else:
        elementDatDir = '/home/oliver/Dropbox/Seidler_Lab/physical_data/elementDensities.csv'
    f = open(elementDatDir, 'r')
    dat = np.genfromtxt(f, dtype = (int, np.dtype('S5'), float), delimiter = ',')
    return dat

def getElementDensity(element): 
    return  filter(lambda x : x[1] == element, getElementDensities())[0]

def getElementName(atomicN): 
    if os.name == 'nt':
        elementNamesf = 'E:\\Dropbox\\Seidler_Lab\\physical_data\\elementNames.csv'
    else:
        elementNamesf = '/home/oliver/Dropbox/Seidler_Lab/physical_data/elementNames.csv'
    elementLabels = np.genfromtxt(elementNamesf, dtype = (np.dtype('S10'), np.dtype('S10')))
    extractedLabel = filter(lambda x : atomicN == int(x[0]), elementLabels)
    if len(extractedLabel) == 0: 
        raise ValueError("Element " + element + "not found. Valid keys \
            are: ")
    name = extractedLabel[0][1]
    return name
   
 

#download K fluorescence branching ratio data for a given element from 
#x-ray periodinc table website
#return a dictionary conatining the branching ratio and energy (in eV)
def Kbranch(element): 
    branchRatioTemplate = 'http://csrri.iit.edu/cgi-bin/period-form?ener=&name=<name>'
    url = replaceAll({'<name>': element}, branchRatioTemplate)
    f = urllib.urlopen(url)
    htmltxt = f.read()
    patt = re.compile('Fluorescence<br>yield</br></caption>\\n <tr><th align=left>K</th><td> *([\.0-9]*).*', re.DOTALL)
    patt2 = re.compile('Edge Energies<br>\(keV\)</br></caption>\\n <tr><th align=left>K</th><td> *([\.0-9]*).*')
    pattFind = re.search(patt, htmltxt)
    eFind = re.search(patt2, htmltxt)
    energy = 1000. * float(eFind.group(1))
    return {"ratio": float(pattFind.group(1)), "energy": energy}



def alphaBetaBranch(element): 
    if os.name == 'nt':
        elementNamesf = 'E:\\Dropbox\\Seidler_Lab\\physical_data\\elementNames.csv'
        dat = np.genfromtxt('E:\\Dropbox\\Seidler_Lab\\physical_data\\transition_metal_branching_ratios.txt', dtype = (float, int,  np.dtype('S4'), float))
    else:
        elementNamesf = '/home/oliver/Dropbox/Seidler_Lab/physical_data/elementNames.csv'
        dat = np.genfromtxt('/home/oliver/Dropbox/Seidler_Lab/physical_data/transition_metal_branching_ratios.txt', dtype = (float, int,  np.dtype('S4'), float))
        
#    elementLabels = np.genfromtxt(elementNamesf, dtype = (np.dtype('S5'), np.dtype('S5')))
    elementLabels = np.genfromtxt(elementNamesf, dtype = (np.dtype('S10'), np.dtype('S10')))
    extractedLabel = filter(lambda x : element == x[1], elementLabels)
    if len(extractedLabel) == 0: 
        raise ValueError("Element " + element + "not found. Valid keys \
            are: ")
    atomicN = int(extractedLabel[0][0])
    fLines = filter(lambda x : x[1] == atomicN, dat)
    return fLines
#    extracted = filter(lambda x : element == x[

def volumeToMassFractions(alldensities, allelements, elements, volFs):
    indices = np.searchsorted(allelements, elements)
    masses = np.array(alldensities)[indices] * np.array(volFs)
    masses = masses/sum(masses)
    return masses
