import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import matplotlib.ticker as ticker
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import wfdb as wf
from wfdb.processing import normalize_bound, resample_singlechan, resample_ann

# also calculate the Heart-Rate in beats per minute(BPM) for given duration
# if there are b beats in a in t sec duration then
# beats per second = b/t
# beats per minute = 60* (b/t)

# Resampled every signal to this rate for consistency
BASIC_SRATE = 128 #Hz

mit_mve_labs_dict = {'(AFIB\x00':0, '(ASYS\x00':1, '(B\x00':2, '(BI\x00':3, '(HGEA\x00':4, '(N\x00':5, '(NSR\x00':5, '(NOD\x00':6, '(NOISE\x00':7, '(PM\x00':8, '(SBR\x00':9, '(SVTA\x00':10, '(VER\x00':11, '(VF\x00':12, '(VFIB\x00':12, '(VFL\x00':13, '(VT\x00':14}

def data_split(iSignal, iRpeaks, iLabels, inRpeaks, inLabels):
    split_data = []
    split_rpeaks = []
    split_rnpeaks = []
    split_labels = []
    split_nlabels = []

    chunk_len = int(10 * BASIC_SRATE)  # 10 seconds in samples
    sig_len = len(iSignal)
    # number of chunks (ceil)
    n_chunks = int(np.ceil(sig_len / chunk_len)) if sig_len > 0 else 0

    for k in range(n_chunks):
        pf = k * chunk_len
        pt = min((k + 1) * chunk_len, sig_len)

        signal_slice = np.asarray(iSignal[pf:pt], dtype=float)
        # pad last chunk if shorter than chunk_len so downstream stacking works
        if signal_slice.shape[0] < chunk_len:
            pad_width = chunk_len - signal_slice.shape[0]
            signal_slice = np.pad(signal_slice, (0, pad_width), mode='constant', constant_values=0.0)

        # find r-peaks and non-beat peaks inside the window
        query_list = np.where((iRpeaks >= pf) & (iRpeaks < pt))[0]
        nquery_list = np.where((inRpeaks >= pf) & (inRpeaks < pt))[0]

        rPeaks = iRpeaks[query_list] - pf
        rLabels = iLabels[query_list]
        rnPeaks = inRpeaks[nquery_list] - pf
        rnLabels = inLabels[nquery_list]

        split_data.append(signal_slice)
        split_labels.append(np.array(rLabels).astype('int'))
        split_labels.append(np.array(rnLabels).astype('int'))

    return (split_data, split_labels)

def load_data(record, atr):
    records = np.loadtxt(f'./datasets/{record}/RECORDS', dtype='str',delimiter="\t")
    sigs = []
    labs = []
    for i in records:
        sigs.append(wf.rdrecord(f'./datasets/{record}/{str(i)}'))
        aux = wf.rdann(f'./datasets/{record}/{str(i)}', atr)
        match record:
            case 'mit-bih-mve':
                for ind,d in enumerate(aux.aux_note):    
                    aux.aux_note[ind] = mit_mve_labs_dict[d]
                    break
            case 'mit-bih-sad': #! This dataset has labels on aux.symbol 
                for ind,d in enumerate(aux.symbol):    
                    #aux.aux_note[ind] = mit_mve_labs_dict[d]
                    pass
                break
            case 'mit-bih-arr': #! This dataset has labels on aux.symbol 
                for ind,d in enumerate(aux.symbol):    
                    #aux.aux_note[ind] = mit_mve_labs_dict[d]
                    pass
                break

        labs.append(aux)
    return (sigs, labs)

def get_data():
    #Load records
    mit_mve_sigs, mit_mve_labs = load_data('mit-bih-mve', 'atr')
    mit_sad_sigs, mit_sad_labs = load_data('mit-bih-sad', 'atr')
    mit_arr_sigs, mit_arr_labs = load_data('mit-bih-arr', 'atr')

    #normalize signals and resample to 128hz
    for ind, item in enumerate(mit_mve_sigs):
        mit_mve_sigs[ind] = normalize_bound(item.p_signal, lb=0, ub=1)
        x = mit_mve_sigs[ind][:,0]
        mit_mve_sigs[ind], mit_mve_labs[ind] = resample_singlechan(x, mit_mve_labs[ind], 250, 128)
    
    for ind, item in enumerate(mit_sad_sigs):
        mit_sad_sigs[ind] = normalize_bound(item.p_signal, lb=0, ub=1)


    for ind, item in enumerate(mit_arr_sigs):
        mit_arr_sigs[ind] = normalize_bound(item.p_signal, lb=0, ub=1)
        mit_mve_sigs[ind] = resample_singlechan(mit_arr_sigs[ind], mit_arr_labs[ind], 360, 128)


    ann= np.loadtxt(f'./datasets/mit-bih-sad/ANNOTATORS', dtype='str',delimiter="\t")
    print(ann)


def split_data():
    pass