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
import pandas as pd
import awkward as ak
from hist import Hist
from Common import *
from Constants import *
from makeTrackConversionLUTs import *
import pickle

""" todo:
    1. Plot basic distribution
    2. Try out TPS
"""

""" Variables in EMTFNtuple
l1t_trk_chi2
l1t_trk_chi2_bend
l1t_trk_chi2_rphi
l1t_trk_chi2_rz
l1t_trk_d0
l1t_trk_d0_bits
l1t_trk_eta
l1t_trk_eta_sector
l1t_trk_genuine
l1t_trk_hit_pattern
l1t_trk_mvaqual
l1t_trk_ndof
l1t_trk_nstub
l1t_trk_pdgid
l1t_trk_phi
l1t_trk_phi_bits
l1t_trk_phi_sector
l1t_trk_pt
l1t_trk_q
l1t_trk_rinv
l1t_trk_rinv_bits
l1t_trk_sim_eta
l1t_trk_sim_phi
l1t_trk_sim_pt
l1t_trk_sim_tp
l1t_trk_tanl
l1t_trk_tanl_bits
l1t_trk_theta
l1t_trk_vx
l1t_trk_vy
l1t_trk_vz
l1t_trk_z0
l1t_trk_z0_bits
"""

BarrelLimits = [181, 160, 140, 110, 0]

class L1Tracks(Module):
    def __init__(self, name="L1Tracks"):
        super().__init__(name)
        self.__bookRate()
        self.__bookStubplots()
        self.progLUTs()
        self.etaphidiv = 1<<(13-8)
        ptshift, ptLUT, self.etashift, etaLUT =  pickle.load(open('trackLUT.pickle', 'rb'))
        self.ptshift = [[int(j) for j in i] for i in ptshift]

        self.ptLUT = ak.values_astype(ak.from_iter(ptLUT), np.int32)
        self.etaLUT = ak.values_astype(ak.from_iter(etaLUT), np.int32)

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith("l1t_trk_"):
                setattr(self, k, event[k])
            if k.startswith("gmt_stub_"):
                setattr(self, k, event[k])
            if k.startswith("gmt_tkmu_"):
                setattr(self, k, event[k])
        self.__convertTrk__()
        self.__convertStub__()
        self.findduplicateTracks()

    def __ptetaLUT__(self, inputarray, shift, LUT):
        rtarray = ak.zeros_like(inputarray)
        lutidx = ak.zeros_like(inputarray)
        bounds = ak.zeros_like(inputarray)
        cnts = ak.num(inputarray)
        for i, j in enumerate(shift):
            inrange = (inputarray > j[0]) & (inputarray<j[1])
            if j[2] < 0:
                bounds = ak.where(inrange, inrange, bounds)
                rtarray=ak.where(inrange, j[4], rtarray)
            else:
                shifted = (inputarray>>j[2]) + j[3]
                flatsft = ak.flatten(shifted)
                lutidx = ak.where(ak.flatten(inrange), flatsft, ak.flatten(lutidx))
                lutidx = ak.unflatten(lutidx, cnts)
        self.l1t_trk_hwPt=ak.unflatten(self.ptLUT[ak.flatten(lutidx)], cnts)
        self.l1t_trk_hwPt=ak.strings_astype(self.l1t_trk_hwPt, np.int64)
        rtarray=ak.unflatten(ak.where(ak.flatten(bounds),
                                                ak.flatten(rtarray),
                                                LUT[ak.flatten(lutidx)]),
                                       cnts)
        rtarray=ak.strings_astype(rtarray, np.int32)
        return rtarray

    def qualCut(self):
        # ap_uint<8> etaAddr = muon.eta() < 0 ? ap_uint<8>(-muon.eta() / 256) : ap_uint<8>((muon.eta()) / 256);
        # ap_uint<8> ptAddr = muon.pt() > 4095 ? ap_uint<8>(15) : ap_uint<8>(muon.pt() / 256);
        # ap_uint<8> addr = ptAddr | (etaAddr << 4);
        # ap_uint<8> qualityCut = lt_tpsID[addr];
        etaAddr = abs(self.l1t_trk_etaI)/256
        ptAddr = ak.where(self.l1t_trk_hwPt > 4095, 15, self.l1t_trk_hwPt / 256)
        npeta = ak.values_astype(ak.flatten(etaAddr), np.int32)
        nppt = ak.values_astype(ak.flatten(ptAddr), np.int32)
        addr = nppt | (npeta << 4)
        l1t_trk_qualCut = ak.from_iter(globals()['lt_tpsID'])[addr]
        self.l1t_trk_qualCut = ak.unflatten(l1t_trk_qualCut, ak.num(self.l1t_trk_etaI))

    def qualTPSID(self, idx):
        pass



    def findduplicateTracks(self):
        df= ak.to_dataframe(ak.zip([self.l1t_trk_rinv_bits,
                                    self.l1t_trk_phi_bits,
                                    self.l1t_trk_tanl_bits]))
        duponly = df.duplicated()
        dff = df[duponly]
        cnt = dff.reset_index(level=1).index.value_counts()
        self.h["trk_dup_bits"].fill(cnt.values)

        df= ak.to_dataframe(ak.zip([self.l1t_trk_hwPt, self.l1t_trk_phiI,
                                    self.l1t_trk_etaI]))
        duponly = df.duplicated()
        dff = df[duponly]
        cnt = dff.reset_index(level=1).index.value_counts()
        self.h["trk_dup_conv"].fill(cnt.values)
        df= ak.to_dataframe(ak.zip([self.l1t_trk_pt, self.l1t_trk_phi,
                                    self.l1t_trk_eta]))
        duponly = df.duplicated()
        dff = df[duponly]
        cnt = dff.reset_index(level=1).index.value_counts()
        self.h["trk_dup_float"].fill(cnt.values)



    def __convertTrk__(self):
        """ Convert tracks for GMT
        """
        BITSTTCURV = 15
        maxCurv_ = 0.00855
        l1t_trk_rinvI = ak.values_astype(self.l1t_trk_rinv * (1 << (BITSTTCURV - 1)) / maxCurv_, np.int32)
        self.l1t_trk_rinvI = l1t_trk_rinvI
        absl1t_trk_rinvI = ak.where(l1t_trk_rinvI > 0, l1t_trk_rinvI, -1* l1t_trk_rinvI)
        self.l1t_trk_hwPt = self.__ptetaLUT__(absl1t_trk_rinvI, self.ptshift, self.ptLUT)

        BITSPHI = 13
        temp = ak.flatten(self.l1t_trk_phi *(1 << (BITSPHI - 1)) / (math.pi))
        self.l1t_trk_phiI = ak.values_astype(ak.unflatten(temp, ak.num(self.l1t_trk_phi)), np.int32)
        etalsb = 2*math.pi/pow(2, 13)
        self.l1t_trk_ptI = ak.values_astype(self.l1t_trk_pt  / 0.0325, np.int32);
        ## Todo : add tanL to the ntuple
        self.l1t_trk_etaI = ak.values_astype(self.l1t_trk_eta  / etalsb, np.int32);
        self.l1t_trk_etaRed = self.l1t_trk_etaI / (1 << (13 - 8))

        ### phi_sector to bit
        self.l1t_trk_roi = (1<<self.l1t_trk_phi_sector)
        self.qualCut()

    def __convertStub__(self):
        self.h["stub_phi1"].fill(ak.flatten(self.gmt_stub_phi1))
        self.h["stub_phi2"].fill(ak.flatten(self.gmt_stub_phi2))
        stub_phisector = []
        idx = ak.local_index(self.gmt_stub_phi1, axis=1)
        self.gmt_stub_phi = ak.where(self.gmt_stub_phi1 == 0,
                                     self.gmt_stub_phi2, self.gmt_stub_phi1)
        self.gmt_stub_roi = ak.zeros_like(self.gmt_stub_phi)
        for i in range(9):
            center = ak.full_like(self.gmt_stub_phi, i*910/32)
            dphi = self.deltaphi(self.gmt_stub_phi,  center) 
            diff = (dphi< 42 ) & (self.gmt_stub_phi != 0)
            # print("ROI", i, dphi[diff][0], self.gmt_stub_phi1[diff][0], self.gmt_stub_phi2[diff][0], self.gmt_stub_eta1[diff][0], self.gmt_stub_eta2[diff][0])
            self.gmt_stub_roi = self.gmt_stub_roi | ((1<<i) * diff)
            stub_phisector.append((i+1)*diff)
        self.gmt_stub_phi_sector = ak.zip(list(stub_phisector))

    def __bookStubplots(self):
        self.h.update({
            "stub_phi1": Hist.new.Reg(91, 0, 910, name="phi1").Double(),
            "stub_phi2": Hist.new.Reg(91, 0, 910, name="phi2").Double(),
            "stub_phi1LSB": Hist.new.Reg(10, 0, 10, name="phi1LSB").Double(),
            "stub_phi2LSB": Hist.new.Reg(10, 0, 10, name="phi1LSB").Double(),
        })

    def __bookRate(self):
        self.h.update({
            "trk_rate" : Hist.new.Reg(100, 0, 100, name="rate").Double(),
            "trk_pt" : Hist.new.Reg(100, 0, 200, name="pt").Double(),
            "trk_chi2" : Hist.new.Reg(100, 0, 100, name="chi2").Double(),
            "trk_dup_bits" : Hist.new.Reg(10, 0, 10, name="trk_dup_bits").Double(),
            "trk_dup_float" : Hist.new.Reg(10, 0, 10, name="trk_dup_float").Double(),
            "trk_dup_conv" : Hist.new.Reg(10, 0, 10, name="trk_dup_conv").Double(),
            # "hb_stacnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Sector Station").Double(),
            # "hb_sta1cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 1").Double(),
        })
        self.h["trk_chi2_eta"] = (Hist.new.Reg(60, -3, 3, name="eta").Reg(100, 0, 200, name="chi2").Double())

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
        self.abseta = abseta
        cnts = ak.num(abseta)
        flateta = ak.values_astype(ak.flatten(abseta), np.int32)

        self.temp1 = [0] * 5
        self.coord1 = [0] * 5
        self.coord2 = [0] * 5
        self.sigma_coord1 = [0]*5
        self.sigma_coord2 = [0]*5
        self.sigma_eta1 = [0]*5
        self.sigma_eta2 = [0]*5

        curv2 = ak.values_astype(self.l1t_trk_rinvI * self.l1t_trk_rinvI /2, np.int64)
        for station in range(5):
            luts = {}
            for k in self.LUTs[station].keys():
                luts[k] = ak.unflatten(self.LUTs[station][k][flateta], cnts)

            self.coord1[station] = (self.l1t_trk_phiI - luts["prop_coord1"] * self.l1t_trk_rinvI / 1024) / self.etaphidiv;
            temp = luts["prop_coord2"] * self.l1t_trk_rinvI / 1024
            ## Todo: BarrelLimits to be checked
            self.coord2[station] = ak.where(abseta < (BarrelLimits[station]), \
                                            -1 * temp /self.etaphidiv , (self.l1t_trk_phiI - temp) /self.etaphidiv)
            self.temp1[station] = (curv2)
            self.sigma_coord1[station] = ((luts["res1_coord1"]  * curv2 ) >> 23) + luts["res0_coord1"]
            self.sigma_coord2[station] = ((luts["res1_coord2"]  * curv2 ) >> 23) + luts["res0_coord2"]
            self.sigma_eta1[station] = ((luts["res1_eta"]  * curv2 ) >> 23 ) + luts["res0_eta1"]
            self.sigma_eta2[station] = ((luts["res1_eta"]  * curv2 ) >> 23 ) + luts["res0_eta2"]

            ## Cap at 4 bits, max at 15 ap_uint<
            self.sigma_coord1[station] = ak.where(self.sigma_coord1[station] > 15, 15, self.sigma_coord1[station])
            self.sigma_coord2[station] = ak.where(self.sigma_coord2[station] > 15, 15, self.sigma_coord2[station])
            self.sigma_eta1[station] = ak.where(self.sigma_eta1[station] > 15, 15, self.sigma_eta1[station])
            self.sigma_eta2[station] = ak.where(self.sigma_eta2[station] > 15, 15, self.sigma_eta2[station])

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

        trkidx=0
        stubidx=0
        qual = 0
        for station in range(5):
            trkidx_, stubidx_, qual_ = self.matching(station)
            if station == 0:
                trkidx = trkidx_
                stubidx = stubidx_
                qual = qual_
            else:
                trkidx = ak.concatenate([trkidx, trkidx_], axis=1)
                stubidx = ak.concatenate([stubidx, stubidx_], axis=1)
                qual = ak.concatenate([qual, qual_], axis=1)

        ## Sort by the same tracks
        ## This should apply to all the trkidx, stubidx and qual
        idx = ak.argsort(trkidx)
        s_trk = trkidx[idx]
        s_stub = stubidx[idx]
        s_qual = qual[idx]
        ## Unique tracks
        muonl1track = GetUnique(s_trk)
        s_trk_len = ak.run_lengths(s_trk)
        # newstubidx = ak.unflatten(stubidx[idx], ak.flatten(s_trk_len), axis=1 )

        ## Get only one stub per TFlayer
        df= ak.to_dataframe({
            "tf": self.gmt_stub_tflayer[s_stub],
            "qual": s_qual,
            "trk": s_trk})
        ## Get min qual per event/track/tf
        df["selqual"] = df.groupby(["entry", "tf", "trk"])['qual'].transform('min')
        df["sel"] = (df.selqual == df.qual)
        ## Selected stub per event/track
        aksel = ak.unflatten(df.sel, ak.num(s_qual))
        trkqual = ak.unflatten(s_qual * aksel, ak.flatten(s_trk_len), axis=1 )
        ## sum of quality per track
        sumqual = ak.sum(trkqual, axis=2)
        passqual = sumqual > self.l1t_trk_qualCut[muonl1track]
        # sell1track = ak.where(passqual, muonl1track, None)
        # print(sell1track)
        # sell1track = ak.drop_none(sell1track)
        # print(sell1track)
        # print(self.l1t_trk_hwPt[sell1track][0])

        # ak.argsort(passqual)
        print("debuging===------")

    def deltaphi(self, phi1, phi2, pi=128):
        ## pi = 128 corredspoding to LSB of 0.00076660156*32, used in Hybrid stub
        diff = abs(phi1 - phi2)
        diff = ak.where(diff > pi, diff - 2*pi , diff)
        diff = ak.where(diff < -1* pi, diff +2*pi , diff)
        return abs(diff)

    def deltaeta(self, eta1, eta2):
        diff = abs(eta1 - eta2)
        return diff

    def matching(self, station):
        argcros = ak.argcartesian([self.gmt_stub_eta1, self.l1t_trk_pt], axis=1)
        stub_idx, trk_idx = ak.unzip(argcros)
        # sel = (self.gmt_stub_tflayer[stub_idx] == station) & (abs(self.l1t_trk_eta[trk_idx]) > 1.4 )
        # sel = (self.gmt_stub_tflayer[stub_idx] == station) & (self.l1t_trk_pt >= 2) & \
        # sel = (self.gmt_stub_tflayer[stub_idx] == station) & (abs(self.l1t_trk_eta[trk_idx]) > 1.4 ) & \
        sel = (self.gmt_stub_tflayer[stub_idx] == station) & \
                ((self.l1t_trk_roi[trk_idx] & self.gmt_stub_roi[stub_idx]) == self.l1t_trk_roi[trk_idx])
                # ((self.l1t_trk_roi[trk_idx] & self.gmt_stub_roi[stub_idx]) == self.l1t_trk_roi[trk_idx]) &\
                # (self.l1t_trk_hwPt[trk_idx] == 79) & \
                # (self.l1t_trk_phiI[trk_idx] == -3671)
                # True
                # (self.l1t_trk_pt[trk_idx] >=2 ) & \

        selstub_idx = stub_idx[sel]
        seltrk_idx = trk_idx[sel]
        print("station", station, "seltrack", self.l1t_trk_phi_sector[seltrk_idx][0], self.l1t_trk_hwPt[seltrk_idx][0],
              self.l1t_trk_phiI[seltrk_idx][0], self.l1t_trk_etaI[seltrk_idx][0], "selstub",
              self.gmt_stub_phi1[selstub_idx][0],
              self.gmt_stub_phi2[selstub_idx][0],
              self.gmt_stub_eta1[selstub_idx][0],
              self.gmt_stub_eta2[selstub_idx][0])

        ## Check phi1
        dphi1 = self.deltaphi(self.coord1[station][seltrk_idx],
                              self.gmt_stub_phi1[selstub_idx])
        dphi2 = self.deltaphi(self.coord2[station][seltrk_idx],
                              self.gmt_stub_phi2[selstub_idx])
        deta1 = self.deltaeta(self.l1t_trk_etaRed[seltrk_idx], 
                              self.gmt_stub_eta1[selstub_idx])
        deta2 = self.deltaeta(self.l1t_trk_etaRed[seltrk_idx], 
                              self.gmt_stub_eta2[selstub_idx])

        ## delta Cap at 4+1 bits, max at 31
        dphi1 = ak.where(dphi1 > 31, 31, dphi1)
        dphi2 = ak.where(dphi2 > 31, 31, dphi2)
        deta1 = ak.where(deta1 > 31, 31, deta1)
        deta2 = ak.where(deta2 > 31, 31, deta2)

        mphi1 = (dphi1 <= self.sigma_coord1[station][seltrk_idx]) & \
                ((self.gmt_stub_qual[selstub_idx] & 0x1) == 0x1)
        mphi2 = (dphi2 <= self.sigma_coord2[station][seltrk_idx]) & \
                ((self.gmt_stub_qual[selstub_idx] & 0x2) == 0x2)
        meta1 = (deta1 <= self.sigma_eta1[station][seltrk_idx]) & \
                ((self.gmt_stub_etaqual[selstub_idx] == 0 ) | ((self.gmt_stub_etaqual[selstub_idx] & 0x1) == 0x1))
        meta2 = (deta2 <= self.sigma_eta2[station][seltrk_idx]) & \
                ((self.gmt_stub_etaqual[selstub_idx] & 0x2) == 0x2)


        # print("index", seltrk_idx, selstub_idx)
        for p1, p2, m1, m2, s1, s2, e1, e2 in zip(dphi1, dphi2, deta1, deta2,
                                          self.sigma_coord1[station][seltrk_idx],
                                          self.sigma_coord2[station][seltrk_idx],
                                          self.sigma_eta1[station][seltrk_idx],
                                          self.sigma_eta2[station][seltrk_idx]):
            # print(station, "delta ", p1, p2, m1, m2, s1, s2)
            print(f"station {station} delta phi1 {p1}+-{s1}, phi2 {p2}+-{s2}, \
                  eta1 {m1}+-{e1} eta2 {m2}+-{e2}")
        for p1, p2, m1, m2 in zip(mphi1, mphi2, meta1, meta2):
            print(station, "match ", p1, p2, m1, m2)

        match1 = mphi1 & meta1
        match2 = mphi2 & meta2
        match3 = mphi1 & (meta1 | meta2) & (self.gmt_stub_etaqual[selstub_idx]==3) & \
                (self.gmt_stub_qual[selstub_idx] ==1)
        endcapmatch = match1 | match2 | match3
        barrelmatch = mphi1 & (meta1 | meta2)
        match = ak.where(abs(self.l1t_trk_eta[seltrk_idx]) < 1.4, barrelmatch, endcapmatch )
        mstub_idx = selstub_idx[match]
        mtrk_idx = seltrk_idx[match]
        ## Start to calculate the quality
        temp = (32 - dphi1)
        print("barrel temp", temp[0])
        barrelquality = temp + (32 - dphi2) *  mphi2
        print("barrelqual", barrelquality[0])
        temp = (32 - dphi1) * (match1 | match3)
        print("endcap temp", temp[0])
        endcapquality = temp +(32- dphi2) * match2
        print("endcap qual", endcapquality[0])
        quality = ak.where(abs(self.l1t_trk_eta[seltrk_idx]) < 1.4, barrelmatch*barrelmatch, endcapquality *endcapmatch )
        print(quality[0])
        quality = ak.values_astype(quality, np.int16)
        print(quality[0])
        print("match in station", station, self.coord1[station][mtrk_idx][0],
              self.gmt_stub_phi1[mstub_idx][0], quality[0])
        return mtrk_idx, mstub_idx, quality[match]

    def printTPS(self):
        pass

    def __fillRate(self):
        self.h["trk_rate"].fill(ak.flatten(self.l1t_trk_pt))
        self.h["trk_pt"].fill(ak.flatten(self.l1t_trk_pt))
        self.h["trk_chi2"].fill(ak.flatten(self.l1t_trk_chi2))
        self.h["trk_chi2_eta"].fill(chi2 = ak.flatten(self.l1t_trk_chi2), eta = ak.flatten(self.l1t_trk_eta))

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()
        # self.propogation()
        # self.matchstubs()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)
