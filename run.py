#!/usr/bin/env python
# encoding: utf-8

# File        : run.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Feb 27
#
# Description : 


import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import pprint
from hist import Hist
from EMTFHits import EMTFHits

import mplhep as hep
hep.style.use("CMS")
# hep.cms.label("Phase 2", data=False, loc=0)

if __name__ == "__main__":
    filelist = ["./test.root"]
    # filelist = ["./test.root", "./test_131.root"]
    filename = [i+":emtfToolsNtupleMaker/tree" for i in filelist]
    events = uproot.iterate(
        # filename(s)
        filename,
        # step_size is still important
        step_size="1 GB",
        # options you would normally pass to uproot.open
        # xrootd_handler=uproot.MultithreadedXRootDSource,
        # num_workers=10,
    )

    hit = EMTFHits()
    for e in events:
        hit.run(e)
    hit.endrun()
