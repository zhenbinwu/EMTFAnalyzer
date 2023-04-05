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
import matplotlib.pyplot as plt
from hist import Hist
from pprint import pprint

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

    def __bookSecCnt(self):
        self.h.update({
            "seccnt" : Hist.new.Reg(20, 0, 20, name="NO. of hits / Sector").Int64(),
            "stationcnt" : Hist.new.Reg(20, 0, 20, name="NO. of hits / Station").Int64(),
        })
        for i in range(1, len(sysnum)):
            self.h["seccnt_type%d" % i]=Hist.new.Reg(20, 0, 20, name="NO. of %s hits / Sector" % sysnum[i]).Int64()

        for i in range(1, 6):
            self.h["stationcnt%d" % i]=Hist.new.Reg(20, 0, 20, name="NO. of hits / Station %d" % i).Int64()
            for j in range(0, len(sysnum)):
                self.h["seccnt_type%d_sta%d" %( j , i) ]= Hist.new.Reg(20, 0, 20, 
                                                                  name="NO. of %s hits / Station %i" % (sysnum[j], i)).Int64()
            
    def __bookExtraHits(self):
        self.h.update({
            "extra_cnt" : Hist.new.Reg(10, 0, 10, name="NO. of extra hits/ Sector").Int64(),
            "extra_station" : Hist.new.Reg(20, 0, 20, name="NO. of hits / Station").Int64(),
        })


    def run(self, event):
        self.__GetEvent__(event)
        self.plotSecCnt()
        self.FindExtraHits()

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

        x = ak.zip({ "sec" : self.hit_endcap * self.hit_sector, "station" : self.hit_station, "sys" : self.hit_subsystem  })
        sorted = x[ak.argsort(x.sec)]
        for i in range(1, 6):
            stats = x.sec[ x.station == i]
            cnt = ak.flatten(ak.run_lengths(stats))
            self.h["stationcnt%d"% i].fill(cnt)
            for j in range(1, len(sysnum)):
                syss = x.sec[ (x.station == i) & (x.sys == j) ]
                cnt =ak.flatten(ak.run_lengths(syss))
                self.h["seccnt_type%d_sta%d" %( j , i) ].fill(cnt)

    def FindExtraHits(self):
        sel = self.hit_emtf_segment > 1
        print(self.hit_subsystem[sel])

    def endrun(self):
        fig = plt.figure(figsize=(10, 8))
        for k in self.h.keys():
            self.h[k].plot()
            # plt.title("df")
            fig.savefig("%s.png" % k)
            fig.clf()
        # self.h["seccnt"].plot()
        # fig.ylabel("test")
