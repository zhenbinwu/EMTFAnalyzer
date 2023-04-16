#!/usr/bin/env python
# encoding: utf-8

# File        : hybrid.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Mar 23
#
# Description : 

import uproot
import numpy as np
import awkward as ak
from hist import Hist
from Common import *

class hbstub():
    def __init__(self, stub=None):
        self.isonly = False
        if stub is not None:
            self.isonly = True
            self.stubs = [stub]
        else:
            self.stubs = list()

    def addstub(self, stub):
        self.stubs.append(stub)

    def phi1(self):
        return min([i.phi for i in self.stubs])

    def theta1(self):
        return min([i.theta1 for i in self.stubs])

    def phi2(self):
        if self.isonly:
            return 0
        else:
            return max([i.phi for i in self.stubs])

    def theta2(self):
        if self.isonly:
            return 0
        else:
            return max([i.theta1 for i in self.stubs])

    def secsta(self):
        secs = np.unique([i.secsta for i in self.stubs])
        if len(secs) > 1:
            print("WEEEEEEE", secs)
        return secs[0]

    def nstubs(self):
        return len(self.stubs)


class HybridStub(Module):
    def __init__(self, name="Hybrids"):
        super().__init__(name)
        # store the hits of this station
        self.stubs = None
        # Cut value
        self.dPhicut = 50
        self.dThetacut = 10
        self.__bookSecCnt()

    def __bookSecCnt(self):
        self.h.update({
            "hb_seccnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Sector").Double(),
            "hb_stacnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Sector Station").Double(),
            "hb_sta1cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 1").Double(),
            "hb_sta2cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 2").Double(),
            "hb_sta3cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 3").Double(),
            "hb_sta4cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 4").Double(),
            "hb_sta5cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 5").Double(),
        })

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith("hit_"):
                setattr(self, k, event[k])

        ## Setting the hybrid stub stations
        isME11 = ((self.hit_emtf_chamber >= 0) & (self.hit_emtf_chamber <= 2)) | \
                ((self.hit_emtf_chamber >= 9) & (self.hit_emtf_chamber <= 11)) 
        isME0 = ((self.hit_emtf_chamber >= 108) & (self.hit_emtf_chamber <= 114))
        isGE11 = ((self.hit_emtf_chamber >= 54) & (self.hit_emtf_chamber <= 56)) | \
                ((self.hit_emtf_chamber >= 63) & (self.hit_emtf_chamber <= 11)) 
        istation1 = isME0 | isME11 | isGE11
        self.hb_station = self.hit_station+1
        self.hb_station = ak.where(istation1, 1, self.hb_station)

        self.stubs = ak.zip({
            "phi" : self.hit_emtf_phi,
            "chamber" : self.hit_emtf_chamber,
            "sector" : self.hit_sector,
            "station" : self.hb_station,
            "theta1" : self.hit_emtf_theta1,
            "theta2" : self.hit_emtf_theta2,
            "secsta" : self.hit_endcap * (self.hit_sector * 10 + self.hb_station),
        })

    def make_equiv_classes(self, pairs):
        groups = {}
        for (x, y) in pairs:
            xset = groups.get(x, set([x]))
            yset = groups.get(y, set([y]))
            jset = xset | yset
            for z in jset:
                groups[z] = jset
        out = set(map(tuple, groups.values()))
        return out

    def RunHb(self):
        ## Global selection
        paircom = ak.combinations(self.stubs, 2, axis=1, fields=["0", "1"])
        pairarg = ak.argcombinations(self.stubs, 2, axis=1)
        diffphi = abs(paircom["0"].phi - paircom["1"].phi)
        difftheta = abs(paircom["0"].theta1 - paircom["1"].theta1)
        sel = (diffphi < self.dPhicut ) & (difftheta < self.dThetacut) & \
                (paircom["0"].secsta == paircom["1"].secsta)
        selarg = pairarg[sel]
        selcom = paircom[sel]

        allhbstubs = []
        self.hb_phi1 = []
        self.hb_phi2 = []
        self.hb_theta1 = []
        self.hb_theta2 = []
        self.hb_secsta = []
        self.hb_nstubs = []
        for i, j in enumerate(selarg):  # Per event
            stubs = self.stubs[i]
            hblist = sorted(self.make_equiv_classes(j.to_list()))
            merged = [ k for item in list(hblist) for k in item]
            alone = list(filter(lambda x : x not in merged, list(range(0, len(stubs)))))

            hbstubs = []
            ## Merging stubs
            for k_ in hblist:
                t = hbstub()
                k = list(k_)
                [t.addstub(stubs[g]) for g in k]
                hbstubs.append(t)
            ## Alone stubs
            [hbstubs.append(hbstub(stubs[ali])) for ali in alone ] 
            allhbstubs.append(hbstubs)
            self.hb_phi1.append([hb.phi1() for hb in hbstubs])
            self.hb_phi2.append([hb.phi2() for hb in hbstubs])
            self.hb_theta1.append([hb.theta1() for hb in hbstubs])
            self.hb_theta2.append([hb.theta2() for hb in hbstubs])
            self.hb_secsta.append([hb.secsta() for hb in hbstubs])
            self.hb_nstubs.append([hb.nstubs() for hb in hbstubs])

        self.hb_phi1 = ak.from_iter(self.hb_phi1)
        self.hb_phi2 = ak.from_iter(self.hb_phi2)
        self.hb_theta1 = ak.from_iter(self.hb_theta1)
        self.hb_theta2 = ak.from_iter(self.hb_theta2)
        self.hb_secsta = ak.from_iter(self.hb_secsta)
        self.hb_nstubs = ak.from_iter(self.hb_nstubs)

    def plotHBcnts(self):
        sortsecsta = ak.sort(self.hb_secsta)
        secstalen = ak.run_lengths(sortsecsta)
        sectors = sortsecsta//10
        # print(self.hb_secsta[1], sortsecsta[1], secstalen[1], sectors[1])
        self.h["hb_stacnt"].fill(ak.flatten(secstalen))
        self.h["hb_seccnt"].fill(ak.flatten(ak.run_lengths(sectors)))
        station = abs(sortsecsta) % 10 * np.sign(sortsecsta)
        # print(sortsecsta[1], np.sign(sortsecsta)[1], station[1])
        # print(ak.run_lengths(sortsecsta[abs(station)==1][1]))
        for i in range(1, 5):
            self.h["hb_sta%dcnt" % i].fill(ak.flatten(ak.run_lengths(sortsecsta[abs(station)==i])))

    def run(self, event):
        self.__GetEvent__(event)
        self.RunHb()
        self.plotHBcnts()

    def endrun(self, outfile, nTotal=0):
        super().endrun(outfile, nTotal)
