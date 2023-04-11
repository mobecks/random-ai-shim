
################################################################
# latest nmrglue package version (09.06.2021) needs this commit https://github.com/jjhelmus/nmrglue.git@6ca36de7af1a2cf109f40bf5afe9c1ce73c9dcdc
################################################################

import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng
import os
import glob
'''
target_def (str) -> how labels are converted
normalize (bool) -> scale data and labels
downsamplefactor (int) -> take every k-th point
ROI_min (int) -> define min. ROI INDEX in [0,32768]. ROI_max = ROI_min + 2048
nr_shims (int) -> Nr. shims
'''
def get_dataset(DATASET_PATH, target_def = 'relative', normalize=True, downsamplefactor=1, ROI_min=None, nr_shims=3):

    norm_factor = 2**15
    shimNames = ["xshim", "yshim","zshim","z2shim", "zxshim", "zyshim", "x2y2shim","xyshim", "z3shim", "z2xshim", "z2yshim","zx2y2shim", "zxyshim", "x3shim", "y3shim"]
    #load data, sort and remove first point
    FNAMEs_shims = glob.glob(DATASET_PATH+"/*/shims.par")
    FNAMEs_shims = sorted(FNAMEs_shims, key = os.path.getmtime)[1:] #sort by creation time
    FNAMEs_1d = glob.glob(DATASET_PATH+"/*/data.1d")
    FNAMEs_1d = sorted(FNAMEs_1d, key = os.path.getmtime)[1:] #sort by creation time
    
    shims_dic = np.array([ { line.split()[0] : int(line.split()[2]) for line in open(f) }
                 for f in FNAMEs_shims])

    DIR_ref_shims = glob.glob(DATASET_PATH+"/RefShims.par")
    ref_shim = np.array([ { line.split()[0] : int(line.split()[2]) for line in open(DIR_ref_shims[0]) }])[0]
    for idx_shim in range(nr_shims,len(shimNames)): # in range of number of relevant shims
            current_name = shimNames[idx_shim]
            ref_shim[current_name] = 0 # added
    
    #dummy scan to get parameters
    dummy_dic,_  = ng.spinsolve.read(os.path.dirname(FNAMEs_1d[0]))

    data_bin = np.empty([len(FNAMEs_1d), int(dummy_dic['spectrum']['xDim']/downsamplefactor) ])
    for idx,f in enumerate(FNAMEs_1d):
        dic, fid = ng.spinsolve.read(os.path.dirname(f))
        # more uniform listing
        udic = ng.spinsolve.guess_udic(dic, fid)
        # fft and phase correction
        spectrum = ng.proc_base.fft(fid)
        spectrum = ng.proc_base.ps(spectrum, p0=float(dic['proc']['p0Phase']) ,p1=float(dic['proc']['p1Phase']))
        spectrum = ng.proc_base.di(spectrum)
        data_bin[idx] = np.array(spectrum)[::downsamplefactor]
        
    xaxis = np.arange(-udic[0]['sw']/2,udic[0]['sw']/2, udic[0]['sw']/udic[0]['size'])
        
    if target_def == 'relative': # return "nr_shims" first order shim values per sample. Relative to (best) reference shim values.
        #create array of labels similar to digital twin (as offset to reference)
        labels = np.empty([len(shims_dic),  nr_shims])
        for idx,d in enumerate(shims_dic):
            for idx_shim in range(nr_shims): # in range of number of relevant shims
                current_name = shimNames[idx_shim]
                labels[idx,idx_shim] = d[current_name] - ref_shim[current_name]       
        if normalize:
            if ROI_min!=None: data_bin = data_bin[:,ROI_min:ROI_min+2048]
            labels = labels / norm_factor
            max_data = 1e5
            data_all = data_bin/max_data          
            return data_all, labels, norm_factor, max_data, xaxis
        else:
            if ROI_min!=None: data_bin = data_bin[:,ROI_min:ROI_min+2048]
            return data_bin, labels, xaxis