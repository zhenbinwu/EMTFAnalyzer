#!/usr/bin/env python
# encoding: utf-8

# File        : hits.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Mar 11
#
# Description : 

import uproot
import numpy as np
import awkward as ak
from hist import Hist
from pprint import pprint
from EMTFLUT import *

sysnum = ['DT', 'CSC', 'RPC', 'GEM', 'ME0']

class EMTFHits():
    def __init__(self):
        # self.prod = prod
        self.h = {}
        self.__bookSecCnt()

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

    def __bookSecCnt(self):
        self.h.update({
            "seccnt" : Hist.new.Reg(20, 0, 20, name="NO. of hits / Sector").Double(),
            "stationcnt" : Hist.new.Reg(20, 0, 20, name="NO. of hits / Station").Double(),
        })
        for i in range(1, len(sysnum)):
            self.h["seccnt_type%d" % i]=Hist.new.Reg(20, 0, 20, name="NO. of %s hits / Sector" % sysnum[i]).Double()

        for i in range(1, 6):
            self.h["stationcnt%d" % i]=Hist.new.Reg(20, 0, 20, name="NO. of hits / Station %d" % i).Double()
            for j in range(0, len(sysnum)):
                self.h["seccnt_type%d_sta%d" %( j , i) ]= Hist.new.Reg(20, 0, 20, 
                                                                  name="NO. of %s hits / Station %i" % (sysnum[j], i)).Double()
        # for i in EMTFSiteMap.keys():
            # self.h["phidist_%s" % EMTFSiteMap[i]] = Hist.new.Reg(60, -30, 30,
                                                                 # name="phi vs offline phi").Double()
            
    def __bookExtraHits(self):
        self.h.update({
            "extra_cnt" : Hist.new.Reg(10, 0, 10, name="NO. of extra hits/ Sector").Double(),
            "extra_station" : Hist.new.Reg(20, 0, 20, name="NO. of hits / Station").Double(),
        })

    def run(self, event):
        self.__GetEvent__(event)
        self.plotSecCnt()
        self.FindExtraHits()
        # self.StudyResolution()

    def plotSecCnt(self):
        ### Plot per sector
        x = ak.zip({ "sec" : self.hit_endcap*self.hit_sector, "sys" : self.hit_subsystem })
        sorted = x[ak.argsort(x.sec)]
        cnt = ak.run_lengths(sorted.sec)
        self.h["seccnt"].fill(ak.flatten(cnt))
        for i in range(1, len(sysnum)):
            self.h["seccnt_type%d" % i].fill(ak.flatten(ak.run_lengths(sorted.sec[sorted.sys == i])))

        ### Plot per sector/sation
        x = ak.zip({ "sec" : self.hit_endcap*self.hit_sector, "sta" : self.hit_station })
        sorted = x[ak.argsort(x.sec)]
        cnt = ak.run_lengths(sorted.sta)
        self.h["stationcnt"].fill(ak.flatten(cnt))

        x = ak.zip({ "sec" : self.hit_endcap * self.hit_sector, "station" : self.hb_station, "sys" : self.hit_subsystem  })
        sorted = x[ak.argsort(x.sec)]
        for i in range(1, 6):
            stats = x.sec[ x.station == i]
            cnt = ak.flatten(ak.run_lengths(stats))
            self.h["stationcnt%d"% i].fill(cnt)
            for j in range(1, len(sysnum)):
                syss = x.sec[ (x.station == i) & (x.sys == j) ]
                cnt =ak.flatten(ak.run_lengths(syss))
                self.h["seccnt_type%d_sta%d" %( j , i) ].fill(cnt)

    def StudyResolution(self):
        secedge = (15 + self.hit_endcap * 60)
        secedge = ak.where(secedge > 180, secedge-360, secedge)
        # print(self.hit_glob_phi[0], secedge[0],  (self.hit_glob_phi - secedge)[0] )
        phidiff = self.hit_emtf_phi * phiLSB - self.hit_glob_phi
        print(self.hit_emtf_phi[0], self.hit_glob_phi[0],
              (self.hit_endcap*self.hit_sector)[0], (self.hit_emtf_phi * phiLSB)[0] )
        for i in EMTFSiteMap.keys():
            self.h["phidist_%s" % EMTFSiteMap[i]].fill(ak.flatten(phidiff[self.hit_emtf_site == i]))


    def FindExtraHits(self):
        sel = self.hit_emtf_segment > 1
        # print(self.hit_subsystem[sel])

    def endrun(self, outfile):
        for k in self.h:
            outfile["EMTFHits/%s" % k] = self.h[k]
