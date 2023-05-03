#!/usr/bin/env python
# encoding: utf-8

# File        : hybrid.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Mar 23
#
# Description : 

import uproot
import math
import numpy as np
import awkward as ak
from hist import Hist
from Common import *
from Constants import *

""" Variables in EMTFNtuple
l1t_trk_chi2
l1t_trk_eta
l1t_trk_eta_sector
l1t_trk_genuine
l1t_trk_hit_pattern
l1t_trk_ndof
l1t_trk_pdgid
l1t_trk_phi
l1t_trk_phi_sector
l1t_trk_pt
l1t_trk_q
l1t_trk_rinv
l1t_trk_sim_eta
l1t_trk_sim_phi
l1t_trk_sim_pt
l1t_trk_sim_tp
l1t_trk_theta
l1t_trk_vx
l1t_trk_vy
l1t_trk_vz
"""

class L1Tracks(Module):
    def __init__(self, name="L1Tracks"):
        super().__init__(name)
        self.__bookRate()
        self.progLUTs()
        self.etaphidiv = 1<<(13-8)

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith("l1t_trk_"):
                setattr(self, k, event[k])
            if k.startswith("gmt_stub_"):
                setattr(self, k, event[k])
        BITSTTCURV = 15
        maxCurv_ = 0.00855
        self.l1t_trk_rinvI = ak.values_astype(self.l1t_trk_rinv * (1 << (BITSTTCURV - 1)) / maxCurv_, np.int32)
        BITSPHI = 13
        self.l1t_trk_phiI =  ak.values_astype(self.l1t_trk_phi *(1 << (BITSPHI - 1)) / (math.pi), np.int32)
        etalsb = 2*math.pi/pow(2, 13)
        self.l1t_trk_etaI = ak.values_astype(self.l1t_trk_eta  / etalsb, np.int32);


    def __bookRate(self):
        self.h.update({
            "trk_rate" : Hist.new.Reg(100, 0, 100, name="rate").Double(),
            # "hb_stacnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Sector Station").Double(),
            # "hb_sta1cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 1").Double(),
        })

    def progLUTs(self):
        self.LUTs = []
        nstation = 5
        for i in range(nstation):
            lutmap = {
                "prop_coord1" : ak.from_iter(globals()['lt_prop_coord1_%d' % i]),
                "prop_coord2" : ak.from_iter(globals()['lt_prop_coord2_%d' % i]),
                "res0_coord1" : ak.from_iter(globals()['lt_res0_coord1_%d' % i]),
                "res1_coord1" : ak.from_iter(globals()['lt_res1_coord1_%d' % i]),
                "res0_coord2" : ak.from_iter(globals()['lt_res0_coord2_%d' % i]),
                "res1_coord2" : ak.from_iter(globals()['lt_res1_coord2_%d' % i]),
                "res0_eta1" : ak.from_iter(globals()['lt_res0_eta1_%d' % i]),
                "res1_eta" : ak.from_iter(globals()['lt_res1_eta_%d' % i]),
                "res0_eta2" : ak.from_iter(globals()['lt_res0_eta2_%d' % i]),
            }
            self.LUTs.append(lutmap)

    def propogation(self):
        abseta = abs(self.l1t_trk_etaI)/8
        cnts = ak.num(abseta)
        flateta = ak.values_astype(ak.flatten(abseta), np.int32)

        self.coord1 = [0] * 5
        self.coord2 = [0] * 5
        self.sigma_coord1 = [0]*5 
        self.sigma_coord2 = [0]*5 
        self.sigma_eta1 = [0]*5
        self.sigma_eta2 = [0]*5

        for station in range(5):
            luts = {}
            for k in self.LUTs[station].keys():
                luts[k] = ak.unflatten(self.LUTs[station][k][flateta], cnts)

            curv2 = ak.values_astype(self.l1t_trk_rinvI * self.l1t_trk_rinvI /2, np.int64)
            self.coord1[station] = (self.l1t_trk_phiI - luts["prop_coord1"] * self.l1t_trk_rinvI / 1024) / self.etaphidiv;
            self.coord2[station] = (self.l1t_trk_phiI - luts["prop_coord2"] * self.l1t_trk_rinvI / 1024) /self.etaphidiv;
            self.sigma_coord1[station] = (luts["res1_coord1"]  * curv2 ) >> 23 + luts["res0_coord1"] 
            self.sigma_coord2[station] = (luts["res1_coord2"]  * curv2 ) >> 23 + luts["res0_coord2"] 
            self.sigma_eta1[station] = (luts["res1_eta"]  * curv2 ) >> 23 + luts["res0_eta1"] 
            self.sigma_eta2[station] = (luts["res1_eta"]  * curv2 ) >> 23 + luts["res0_eta2"] 

    def matchstubs(self):
        self.stubs = ak.zip({
            "phi1" : self.gmt_stub_phi1,
            "phi2" : self.gmt_stub_phi2,
            "eta1" : self.gmt_stub_eta1,
            "eta2" : self.gmt_stub_eta2,
            "layer": self.gmt_stub_tflayer,
        })

        self.props = ak.zip( {
                "coord1"                     : self.coord1,
                "coord2"                     : self.coord2,
                "sigma_coord1"               : self.sigma_coord1,
                "sigma_coord2"               : self.sigma_coord2,
                "sigma_eta1"                 : self.sigma_eta1,
                "sigma_eta2"                 : self.sigma_eta2,
            })

        station = 0
        self.matching(station)

    def deltaphi(self, phi1, phi2):
        diff = phi1 - phi2
        return diff

    def deltaeta(self, eta1, eta2):
        diff = eta1 - eta2
        return diff

    def matching(self, station):
        argcros = ak.argcartesian([self.gmt_stub_eta1, self.l1t_trk_pt], axis=1)
        stub_idx, trk_idx = ak.unzip(argcros)
        sel = self.gmt_stub_tflayer[stub_idx] == station
        selstub_idx = stub_idx[sel]
        seltrk_idx = trk_idx[sel]

        ## Check phi1
        dphi1 = self.deltaphi(self.coord1[station][seltrk_idx], self.gmt_stub_phi1[selstub_idx])
        dphi2 = self.deltaphi(self.coord2[station][seltrk_idx], self.gmt_stub_phi2[selstub_idx])
        deta1 = self.deltaeta(self.l1t_trk_etaI[seltrk_idx], self.gmt_stub_eta1[selstub_idx])
        deta2 = self.deltaeta(self.l1t_trk_etaI[seltrk_idx], self.gmt_stub_eta2[selstub_idx])
        mphi1 = dphi1 <= self.sigma_coord1[station][seltrk_idx]
        mphi2 = dphi2 <= self.sigma_coord2[station][seltrk_idx]
        meta1 = deta1 <= self.sigma_eta1[station][seltrk_idx]
        meta2 = deta2 <= self.sigma_eta2[station][seltrk_idx]
        match1 = mphi1 & meta1
        match2 = mphi2 & meta2
        match3 = mphi1 & (meta1 | meta2)
        match = match1 | match2 | match3
        mstub_idx = selstub_idx[match]
        mtrk_idx = seltrk_idx[match]

    def __fillRate(self):
        self.h["trk_rate"].fill(ak.flatten(self.l1t_trk_pt))

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()
        self.propogation()
        self.matchstubs()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)
