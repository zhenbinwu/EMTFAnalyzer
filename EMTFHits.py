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
import hist
from pprint import pprint
from Common import *

""" todo:
"""

""" Variables in EMTFNtuple
hit_bend
hit_bx
hit_chamber
hit_cscfr
hit_cscid
hit_emtf_bend
hit_emtf_chamber
hit_emtf_host
hit_emtf_phi
hit_emtf_qual1
hit_emtf_qual2
hit_emtf_segment
hit_emtf_site
hit_emtf_theta1
hit_emtf_theta2
hit_emtf_time
hit_emtf_timezones
hit_emtf_zones
hit_endcap
hit_glob_perp
hit_glob_phi
hit_glob_theta
hit_glob_time
hit_glob_z
hit_id
hit_layer
hit_neighbor
hit_pattern
hit_quality
hit_ring
hit_sector
hit_sim_tp1
hit_sim_tp2
hit_station
hit_strip
hit_strip_hi
hit_strip_lo
hit_subbx
hit_subsector
hit_subsystem
hit_valid
hit_wire1
hit_wire2
""" 

class EMTFHits(Module):
    def __init__(self, name="EMTFHits"):
        super().__init__(name)
        self.__bookSecCnt()

    def __GetEvent__(self, event):
        super().__GetEvent__(event)
        for k in dir(event):
            if k.startswith("hit_"):
                setattr(self, k, event[k])

    def __bookSecCnt(self):
        self.h.update({
            "CSC_Bending" : Hist.new.Reg(20, -10, 10, name="Bending angle of CSC;Bending;CSC Hits").Double(),
            "NStation" : Hist.new.Reg(10, 0, 10, name="NO of EMTF Hit Station;X;Y").Double(),
            "NHBLayer" : Hist.new.Reg(10, 0, 10, name="NO of Hybrid Stub Layer").Double(),
            "NTFLayer" : Hist.new.Reg(10, 0, 10, name="NO of TF Layer").Double(),
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
        for i in EMTFSiteMap.keys():
            self.h["phidist_%s" % EMTFSiteMap[i]] = Hist.new.Reg(60, -30, 30,
                                                                 name="phi vs offline phi").Double()
        self.h["station_map"] = (
            Hist.new
            .Reg(250, 500, 1000, name="x", label ="Z")
            .Reg(180, 0, 180, name="y", label="theta")
            .Reg(10, 0, 10, name="z", label="TFLayer")
            .Double()
        )
        self.h["hit_thetaVsLayer"] = (
            Hist.new
            .Reg(360, 0, 360, name="x", label="theta")
            .Reg(5, 0.5, 5.5, name="y", label="TFLayer")
            .Double()
        )
        self.h["hit_siteVsBend"] = (
            Hist.new
            .Reg(12, 0, 12, name="x", label="site")
            .Reg(240, 120, -120, name="y", label="bend")
            .Double()
        )

            
    def __bookExtraHits(self):
        self.h.update({
            "extra_cnt" : Hist.new.Reg(10, 0, 10, name="NO. of extra hits/ Sector").Double(),
            "extra_station" : Hist.new.Reg(20, 0, 20, name="NO. of hits / Station").Double(),
        })

    def run(self, event):
        super().run(event)
        self.plotSecCnt()
        self.FindExtraHits()
        self.StudyResolution()
        self.plotTFLayer()

    def plotTFLayer(self):
        self.h["station_map"].fill(x=ak.flatten(abs(self.hit_glob_z)),
                                   y=ak.flatten(self.hit_glob_theta), 
                                   z=ak.flatten(self.hb_tflayer))
        self.h["hit_thetaVsLayer"].fill( x=ak.flatten(self.hit_glob_theta), 
                                   y=ak.flatten(self.hb_layer))
        self.h["hit_siteVsBend"].fill( x=ak.flatten(self.hit_emtf_site), 
                                   y=ak.flatten(self.hit_bend))
        self.h["CSC_Bending"].fill(ak.flatten(self.hit_bend[self.hit_emtf_site < 5]))
    
    
    def sortsplit(self, input, variable):
        sorted = input[ak.argsort(input[variable])]
        output = ak.unflatten(ak.flatten(sorted), ak.flatten(ak.run_lengths(sorted[variable])))
        return output 


    def plotSecCnt(self):
        ### Plot per sector/sation
        x = ak.zip({ "sec" : self.hit_endcap*self.hit_sector, "sta" : self.hit_station })
        sorted = x[ak.argsort(x.sec)]
        cnt = ak.run_lengths(sorted.sta)
        self.h["stationcnt"].fill(ak.flatten(cnt))

        ### Create a large Zip
        seczip = ak.zip({ "sector" : self.hit_endcap * self.hit_sector, 
                         "hitstation" : self.hit_station, 
                         "hblayer" : self.hb_station, 
                         "tflayer" : self.hb_tflayer, 
                         "subsys" : self.hit_subsystem,
                         "site" : self.hit_emtf_site,
                         "host" : self.hit_emtf_host
                        })
        ### Sorted per sector
        sortsec = self.sortsplit(seczip, "sector")
        ### Plot subsystem per sector
        cnt = ak.run_lengths(sortsec.sector)
        self.h["seccnt"].fill(ak.flatten(cnt))
        sec_subsys = sortsec[ak.argsort(sortsec.subsys)]
        for i in range(1, len(sysnum)):
            self.h["seccnt_type%d" % i].fill(ak.flatten(ak.run_lengths(sec_subsys.sector[sec_subsys.subsys == i])))
        ### Plot per station/sector
        # for i in range(1, 6):
            # stats = x.sec[x.station == i]
            # cnt = ak.flatten(ak.run_lengths(stats))
            # self.h["stationcnt%d"% i].fill(cnt)
            # for j in range(1, len(sysnum)):
                # syss = x.sec[ (x.station == i) & (x.sys == j) ]
                # cnt =ak.flatten(ak.run_lengths(syss))
                # self.h["seccnt_type%d_sta%d" %( j , i) ].fill(cnt)

    def StudyResolution(self):
        secedge = (15 + self.hit_endcap * 60)
        secedge = ak.where(secedge > 180, secedge-360, secedge)
        phidiff = self.hit_emtf_phi * phiLSB - self.hit_glob_phi
        print(self.hit_emtf_phi[0], self.hit_glob_phi[0],
              (self.hit_endcap*self.hit_sector)[0], (self.hit_emtf_phi * phiLSB)[0] )
        for i in EMTFSiteMap.keys():
            self.h["phidist_%s" % EMTFSiteMap[i]].fill(ak.flatten(phidiff[self.hit_emtf_site == i]))


    def FindExtraHits(self):
        sel = self.hit_emtf_segment > 1
        # print(self.hit_subsystem[sel])

    def endrun(self, outfile, nTotal=0):
        for i in range(5):
            self.h["station_map%d" % i]  = self.h["station_map"][:, :, hist.loc(i)]
        super().endrun(outfile, nTotal)
