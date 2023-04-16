#!/usr/bin/env python
# encoding: utf-8

# File        : Common.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Apr 14
#
# Description : 

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

sysnum = ['DT', 'CSC', 'RPC', 'GEM', 'ME0']
LHCnBunches = 2760
LHCFreq = 11.246 #kHz

phiLSB = 0.016666 


class Module():
    def __init__(self, name):
        # storing histogram
        self.h = {}
        self.folder = name

    def __GetEvent__(self, event):
        pass

    def run(self, event):
        self.__GetEvent__(event)

    def endrun(self, outfile, nTotal=0):
        for k in self.h:
            if "rate" in k:
                self.h[k] = ConvertRate(self.h[k], nTotal)
            outfile["%s/%s" % (self.folder, k)] = self.h[k]

def ConvertRate(hist, nZB=0):
    if nZB == 0:
        return hist
    content = hist.values()
    newcontent = np.flip(np.cumsum(np.flip(content))) * LHCnBunches * LHCFreq / nZB
    hist[...] = newcontent
    return hist
