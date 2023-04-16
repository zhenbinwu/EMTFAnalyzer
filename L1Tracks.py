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

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith("l1t_trk_"):
                setattr(self, k, event[k])

    def __bookRate(self):
        self.h.update({
            "trk_rate" : Hist.new.Reg(100, 0, 100, name="rate").Double(),
            # "hb_stacnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Sector Station").Double(),
            # "hb_sta1cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 1").Double(),
        })

    def __fillRate(self):
        self.h["trk_rate"].fill(ak.flatten(self.l1t_trk_pt))

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)
