#!/usr/bin/env python3
# encoding: utf-8

# File        : run.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2023 Feb 27
#
# Description : 

import os
import uproot
import argparse
import numpy as np
import awkward as ak
import pprint
import subprocess
from hist import Hist
from EMTFHits import EMTFHits
from Hybrid import HybridStub
from TrackerMuons import TrackerMuons
from L1Tracks import L1Tracks

import mplhep as hep
hep.style.use("CMS")
# hep.cms.label("Phase 2", data=False, loc=0)
eosfolder = '/eos/uscms/store/user/benwu/EMTF_Ntupler/Fall22_v1/'

samplemap = {
    "DsToTauTo3Mu":  "DsToTauTo3Mu_TuneCP5_14TeV-pythia8",
    "DYToLL"      :  "DYToLL_M-50_TuneCP5_14TeV-pythia8",
    "MinBias"     :  "MinBias_TuneCP5_14TeV-pythia8",
    "Muon200"     :  "SingleMuon_Pt-0To200_Eta-1p4To3p1-gun",
    "Muon500"     :  "SingleMuon_Pt-200To500_Eta-1p4To3p1-gun",
    "TauTo3Mu"    :  "TauTo3Mu_TuneCP5_14TeV-pythia8",
    "TTTo2L2Nu"   :  "TTTo2L2Nu_TuneCP5_14TeV-powheg-pythia8",
    "TTToSemi"    :  "TTToSemiLepton_TuneCP5_14TeV-powheg-pythia8"
}

def eosls(sample):
    hostname =os.uname().nodename
    if sample not in samplemap.keys():
        return None
    if "local" in hostname:
        outputLocation = "/Users/benwu/Work/Data/Fall22_GMT_v2/" + sample
    else:
        outputLocation = eosfolder + samplemap[sample]
    if "eos" in outputLocation:
        p = subprocess.Popen("eos root://cmseos.fnal.gov find %s -type f -name \'*.root\' " % outputLocation,
                                      stdout=subprocess.PIPE, shell=True)
        (dummyFiles_, _) = p.communicate()
        dummyFiles = str(dummyFiles_, 'UTF-8').split("\n")
        dummyFiles = [ f for f in dummyFiles if f.endswith(".root") ]
    else:
        dummyFiles = [ outputLocation +"/"+i for i in os.listdir(outputLocation)]
    return dummyFiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Running over EMTF Ntuple')
    parser.add_argument("-s", "--sample", dest="sample", default='test',
                        help="sample")
    args = parser.parse_args()

    if args.sample == "test":
        filelist = ["./GMT_sample.root"]
    else:
        filelist = eosls(args.sample)
    filename = [i+":emtfToolsNtupleMaker/tree" for i in filelist]
    events = uproot.iterate(
        filename,
        # step_size is still important
        step_size="2 GB",
        # options you would normally pass to uproot.open
        # xrootd_handler=uproot.MultithreadedXRootDSource,
        # num_workers=10,
    )

    ### Store the output file
    outfile = uproot.recreate("%s.root" % args.sample )

    mod_hit = EMTFHits()
    mod_hbstub = HybridStub()
    mod_tkmuons = TrackerMuons()
    mod_l1trks = L1Tracks()
    modules = [
        mod_hit,
        mod_hbstub,
        mod_tkmuons,
        mod_l1trks,
    ]

    ### Running over samples
    nTotal = 0
    for e in events:
        nEvent = len(e)
        print("Processing %d events" % nEvent)
        nTotal += nEvent
        [m.run(e) for m in modules]

    ### End of run
    if "MB" not in args.sample :
        nTotal = 0
    [m.endrun(outfile, nTotal) for m in modules]
