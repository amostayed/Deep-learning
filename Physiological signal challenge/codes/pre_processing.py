import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.signal as sig
import pywt
import os
import glob
import itertools
#
def ecg_preprocessing(data, wfun, dcmp_levels, chop_levels):
    #
    dcmp_levels = min(dcmp_levels, pywt.dwt_max_level(data.shape[1], pywt.Wavelet(wfun)))
    coeffs = pywt.wavedec(data, wfun, mode='symmetric', level = dcmp_levels, axis = -1)
    #
    coeffs_m = [np.zeros_like(coeffs[idx]) if idx >= -chop_levels  else coeffs[idx] for idx in range(-dcmp_levels- 1, 0)]
    #
    data_recon = pywt.waverec(coeffs_m, wfun, mode='symmetric', axis = -1)
    #
    data_recon = butterworth_high_pass(data_recon, cut_off = 2, order = 1, sampling_freq = 500)
    #
    data_recon = butterworth_notch(data_recon, cut_off = [49, 51], order = 2, sampling_freq = 500)
    #
    return data_recon
#
def butterworth_high_pass(x, cut_off, order, sampling_freq):
    #
    nyq_freq = sampling_freq / 2
    digital_cutoff = cut_off / nyq_freq
    #
    b, a = sig.butter(order, digital_cutoff, btype='highpass')
    y = sig.lfilter(b, a, x, axis = -1)
    #
    return y
#
def butterworth_notch(x, cut_off, order, sampling_freq):
    #
    cut_off = np.array(cut_off)
    nyq_freq = sampling_freq / 2
    digital_cutoff = cut_off / nyq_freq
    #
    b, a = sig.butter(order, digital_cutoff, btype='bandstop')
    y = sig.lfilter(b, a, x, axis = -1)
    #
    return y
#