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
gmt_tkmu_charge
gmt_tkmu_hwbeta
gmt_tkmu_hwd0
gmt_tkmu_hweta
gmt_tkmu_hwiso
gmt_tkmu_hwisosum
gmt_tkmu_hwisosumap
gmt_tkmu_hwphi
gmt_tkmu_hwpt
gmt_tkmu_hwqual
gmt_tkmu_hwz0
gmt_tkmu_phcharge
gmt_tkmu_phd0
gmt_tkmu_pheta
gmt_tkmu_phphi
gmt_tkmu_phpt
gmt_tkmu_phz0
""" 


class TrackerMuons(Module):
    def __init__(self, name="TkMuons"):
        super().__init__(name)
        self.__bookRate()

    def __GetEvent__(self, event):
        for k in dir(event):
            if k.startswith("gmt_tkmu_"):
                setattr(self, k, event[k])

    def __bookRate(self):
        self.h.update({
            "tkmu_rate" : Hist.new.Reg(100, 0, 100, name="rate").Double(),
            # "hb_stacnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Sector Station").Double(),
            # "hb_sta1cnt" : Hist.new.Reg(20, 0, 20, name="NO. of hybrid stubs / Station 1").Double(),
        })

    def __fillRate(self):
        self.h["tkmu_rate"].fill(ak.flatten(self.gmt_tkmu_phpt))

    def run(self, event):
        self.__GetEvent__(event)
        self.__fillRate()

    def endrun(self, outfile, nZB=0):
        super().endrun(outfile, nZB)
