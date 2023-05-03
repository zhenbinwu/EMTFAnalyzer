#!/usr/bin/env python
# encoding: utf-8

# File        : Common.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Apr 14
#
# Description : 

import numpy as np
import awkward as ak
from hist import Hist

EMTFSiteMap = {
     0  : "ME11",
     1  : "ME12,3",
     2  : "ME21,2",
     3  : "ME31,2",
     4  : "ME41,2",
     5  : "RE12,3",
     6  : "RE22,3",
     7  : "RE31,2,3",
     8  : "RE41,2,3",
     9  : "GE11",
     10 : "GE21",
     11 : "ME0",
}


TFLayer = {
    0  : 0, #"ME11",
    1  : 4, #"ME12,3",
    2  : 2, #"ME21,2",
    3  : 1, #"ME31,2",
    4  : 3, #"ME41,2",
    5  : 4, #"RE12,3",
    6  : 2, #"RE22,3",
    7  : 1, #"RE31,2,3",
    8  : 3, #"RE41,2,3",
    9  : -1, #"GE11",
    10 : -1, #"GE21",
    11 : -1, #"ME0",
}

sysnum = ['DT', 'CSC', 'RPC', 'GEM', 'ME0']
LHCnBunches = 2760
LHCFreq = 11.246 #kHz

phiLSB = 0.016666 

def listLUT(ilist, target):
    LUT = ak.from_iter(ilist)
    cnts= ak.num(target)
    out = LUT[ak.flatten(target)]
    return ak.unflatten(out, cnts)

class Module():
    def __init__(self, name):
        # storing histogram
        self.h = { "Nevent" : Hist.new.Reg(2, 0, 2, name="NO. of Events").Double()}
        self.folder = name


    def __GetEvent__(self, event):
        pass

    def run(self, event):
        self.h["Nevent"].fill(event.evt_run > 0)
        self.__GetEvent__(event)

    def endrun(self, outfile, nTotal=0):
        orgkeys = list(self.h.keys())
        for k in orgkeys:
            if "rate" in k:
                self.h["%s_scaled" % k] = ConvertRate(self.h[k], nTotal)
        for k in self.h.keys():
            outfile["%s/%s" % (self.folder, k)] = self.h[k]

def ConvertRate(hist, nZB=0):
    if nZB == 0:
        return hist
    content = hist.values()
    newcontent = np.flip(np.cumsum(np.flip(content))) * LHCnBunches * LHCFreq / nZB
    hist[...] = newcontent
    return hist
