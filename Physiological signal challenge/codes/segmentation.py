import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.signal as sig
import pywt
import os
import glob
import itertools
import pandas as pd
import re
#
from codes.pre_processing import ecg_preprocessing
#
EPS = np.finfo(float).eps
#
def wavelet_filtering(data, wfun, max_level = 8):
    #
    padsize = int((2 ** max_level) * np.ceil(data.shape[0] / (2 ** max_level)) - data.shape[0])
    #
    data_padded = np.pad(data, (0, padsize), 'constant', constant_values=(0, 0))
    #
    wave = pywt.swt(data_padded, wfun, level = max_level, start_level = 0, axis=0)
    #
    wave_m = [[np.zeros((data_padded.shape[0],), dtype = float) for j in range(2)] for i in range(max_level)]  #list
    wave_m[-4][1] = wave[-4][1]; wave_m[-5][1] = wave[-5][1]
    wave_m = [tuple(wave_m[i]) for i in range(max_level)]
    #
    data_const = pywt.iswt(wave_m, wfun)
    if padsize != 0:
        data_const = data_const[:-padsize]
    #
    return data_const
 #
def local_energy(data, window_size):
    #
    energy = sig.convolve(data ** 2, np.ones((window_size,), dtype = 'float')/window_size, mode='same')
    return energy
#
def find_threshold(feature, window_size_for_threshold):
    #
    window_size = window_size_for_threshold
    loc = np.arange(0, len(feature), window_size)
    LM = [np.max(feature[l : min(l + window_size, len(feature))]) for l in loc]
    lm = [np.min(feature[l : min(l + window_size, len(feature))]) for l in loc]
    #
    return np.median([np.median(lm), np.median(LM)])

#
def detect_local_maxima(feature, window_size_for_threshold):
    #
    thresh = find_threshold(feature, window_size_for_threshold)
    #
    y_ = feature * (feature > thresh)
    lobe_detector = y_ != 0
    lobe_detector = lobe_detector.astype(float)
    lobe_detector = np.diff(lobe_detector)
    #
    lobe_edges_left = np.where(lobe_detector == 1)[0]
    lobe_edges_right = np.where(lobe_detector == -1)[0]
    lobe_edges_left = lobe_edges_left
    lobe_edges_right = lobe_edges_right
    #
    if len(lobe_edges_left) == 0 and len(lobe_edges_right) == 1:
        lobe_edges_left = np.array([0])
    #
    if len(lobe_edges_left) == 1 and len(lobe_edges_right) == 0:
        lobe_edges_right = np.array([len(feature) - 1])
    #
    if len(lobe_edges_left) == 0 and len(lobe_edges_right) == 0:
        peak_loc = []
        return peak_loc
    #
    if lobe_edges_left[0] > lobe_edges_right[0]:
        lobe_edges_left =  np.concatenate(([0], lobe_edges_left))
    #
    if lobe_edges_left[-1] > lobe_edges_right[-1]:
        lobe_edges_right =  np.concatenate((lobe_edges_right, [len(feature) - 1]))
    #
    no_of_lobes = len(lobe_edges_left)
    peak_loc = [np.argmax(y_[lobe_edges_left[idx] : lobe_edges_right[idx] + 1]) for idx in range(no_of_lobes)] + lobe_edges_left - 1
    #
    return peak_loc
#
def detect_local_maxima_v2(feature, threshold):
    #
    thresh = threshold
    #
    y_ = feature * (feature > thresh)
    lobe_detector = y_ != 0
    lobe_detector = lobe_detector.astype(float)
    lobe_detector = np.diff(lobe_detector)
    #
    lobe_edges_left = np.where(lobe_detector == 1)[0]
    lobe_edges_right = np.where(lobe_detector == -1)[0]
    lobe_edges_left = lobe_edges_left
    lobe_edges_right = lobe_edges_right
    #
    if len(lobe_edges_left) == 0 and len(lobe_edges_right) == 1:
        lobe_edges_left = np.array([0])
    #
    if len(lobe_edges_left) == 1 and len(lobe_edges_right) == 0:
        lobe_edges_right = np.array([len(feature) - 1])
    #
    if len(lobe_edges_left) == 0 and len(lobe_edges_right) == 0:
        peak_loc = []
        return peak_loc
    #
    if lobe_edges_left[0] > lobe_edges_right[0]:
        lobe_edges_left =  np.concatenate(([0], lobe_edges_left))
    #
    if lobe_edges_left[-1] > lobe_edges_right[-1]:
        lobe_edges_right =  np.concatenate((lobe_edges_right, [len(feature) - 1]))
    #
    no_of_lobes = len(lobe_edges_left)
    peak_loc = [np.argmax(y_[lobe_edges_left[idx] : lobe_edges_right[idx] + 1]) for idx in range(no_of_lobes)]
    peak_loc = peak_loc + lobe_edges_left - 1
    #
    return peak_loc
#
def peak_refine(primary_locations, search_radius, count_threshold):
    #
    primary_locations = np.sort(primary_locations)
    N = len(primary_locations)
    #
    i = 0
    peaks = np.array([])
    while i < N - 1:
        begin = primary_locations[i]
        count = np.sum(np.abs(primary_locations[i:] - begin) < search_radius)
        step = count
        location = np.expand_dims(np.median(primary_locations[i : i + step]), axis = 0)
        if count > count_threshold:
            peaks = np.concatenate((peaks, location))
        i = i + step
    peaks = np.round(peaks).astype(int)
    #
    return peaks
#
def peak_detector_basic(data, wfun, max_level, window_size, window_size_for_threshold, search_radius):
    #
    peak_loc_ensemble = np.array([])
    features = np.zeros_like(data)
    for ch in np.arange(data.shape[0]):
        data_const = wavelet_filtering(data[ch,:], wfun, max_level)
        #
        feature = local_energy(data_const, window_size)
        #
        peak_loc = detect_local_maxima(feature, window_size_for_threshold)
        #
        peak_loc_ensemble = np.concatenate((peak_loc, peak_loc_ensemble))
        features[ch, :] = feature
    #
    count_thres = np.floor(np.sum(np.any(data, axis = 1))/2)
    #
    peaks = peak_refine(peak_loc_ensemble, search_radius, count_thres)
    #
    return peaks, features
#
def retrieve_missing_peaks(peaks, features, missing_thres):
    #
    peaks_proxy = list(peaks)
    #
    if features.shape[1] - peaks[-1] > missing_thres:
        peaks_proxy.append(features.shape[1] - 1)
    #
    if peaks[0] > missing_thres:
        peaks_proxy.insert(0, 0)
    #
    missing_peak_loc = np.where(np.diff(peaks_proxy) > missing_thres)[0]
    missing_peak_loc_plus_1 = np.where(np.diff(peaks_proxy) > missing_thres)[0] + 1
    #
    if len(missing_peak_loc) != 0:
        extra_peaks = np.array([])
        #
        for j in np.arange(len(missing_peak_loc)):
            peak_loc_ens = np.array([])
            for ch in np.arange(features.shape[0]):
                s = int(round((peaks_proxy[missing_peak_loc_plus_1[j]] - peaks_proxy[missing_peak_loc[j]])/6))
                seg = features[ch, peaks_proxy[missing_peak_loc[j]] + s : peaks_proxy[missing_peak_loc_plus_1[j]] - s + 1]
                threshold = features[ch, peaks_proxy[missing_peak_loc[j]]]/8 + features[ch,peaks_proxy[missing_peak_loc_plus_1[j]]]/8;
                np.array(detect_local_maxima_v2(seg, threshold)) + peaks_proxy[missing_peak_loc[j]] + s
                peak_loc_ens = np.concatenate((np.array(detect_local_maxima_v2(seg, threshold)) + peaks_proxy[missing_peak_loc[j]] + s, peak_loc_ens))
        #
            peak_loc_ens = peak_loc_ens.astype(int)
            extra_peaks = np.concatenate((extra_peaks, peak_refine(peak_loc_ens, 100, np.floor(np.sum(np.any(features, axis = 1))/2))))
    #
        peaks = np.concatenate((peaks, extra_peaks))
        peaks = np.sort(peaks).astype(int)
    #
    return peaks
#
def remove_false_peaks(peaks, remove_thres):
    peaks_proxy = list(peaks)
    false_peak_loc = np.where(np.diff(peaks_proxy) < remove_thres)[0] 
    to_delete = np.array([])
    #
    if len(false_peak_loc) != 0:
        #
        for j in np.arange(len(false_peak_loc)):
            #    
            if false_peak_loc[j] == len(peaks_proxy) - 2:
                d1 = peaks_proxy[false_peak_loc[j] - 1] - peaks_proxy[false_peak_loc[j]]
                d2 = peaks_proxy[false_peak_loc[j] - 1] - peaks_proxy[false_peak_loc[j] + 1]  
            else:
                d1 = peaks_proxy[false_peak_loc[j] + 2] - peaks_proxy[false_peak_loc[j]]
                d2 = peaks_proxy[false_peak_loc[j] + 2] - peaks_proxy[false_peak_loc[j] + 1]
            #
            if abs(np.median(np.diff(peaks_proxy)) - d1) > abs(np.median(np.diff(peaks_proxy)) - d2):
                to_delete = np.concatenate((to_delete, [false_peak_loc[j]]))
            else:
                to_delete = np.concatenate((to_delete, [false_peak_loc[j] + 1]))
    #
    peaks = np.delete(peaks, to_delete.astype(int))
    #
    return peaks
#
def peak_detector_with_refinement(file_name, wfun = 'sym8', max_level = 8, window_size = 50, window_size_for_threshold = 1000, search_radius = 50):
    #
    data = sio.loadmat(file_name)['ECG'][0][0][2]
    #
    peaks, features = peak_detector_basic(data, wfun, max_level, window_size, window_size_for_threshold, search_radius)
    #
    missing_peak_thres = 750
    peaks = retrieve_missing_peaks(peaks, features, missing_peak_thres)
    #
    missing_peak_thres = 1.7 * np.median(np.diff(peaks))
    peaks = retrieve_missing_peaks(peaks, features, missing_peak_thres)
    #
    remove_thres = 0.33 * np.median(np.diff(peaks))
    peaks = remove_false_peaks(peaks, remove_thres)
    #
    return peaks, features
#
def set_to_desired_length(array, max_len = 1000):
    #
    if array.shape[1] > max_len:
        array = array[:, 0 : max_len]
    if array.shape[1] < max_len:
        array = np.pad(array, [(0, 0), (0, max_len - array.shape[1])], mode='constant')
    return array
#
def extract_features(peaks):
    #
    return np.diff(peaks)
#
def extract_ecg_segments(peaks, data, data_label, max_length):
    #
    # read data as float32, labels as int32, sequence length as int32
    data = data.astype(np.float32)
    # clean the data first
    data = ecg_preprocessing(data, 'sym8', 8, 3)
    #
    num_of_peaks = len(peaks)
    no_channels = 12
    data_length = np.expand_dims(data.shape[1], axis = 0)
    #
    if num_of_peaks == 5:
        peak_to_peak = np.concatenate([np.diff(peaks), data_length - peaks[-1]])
    else:
        peak_to_peak = np.diff(peaks)
    #
    #
    if num_of_peaks <= 4:
        ecg_segments = np.expand_dims(set_to_desired_length(data, max_len = max_length), axis = 0) # Rank-3 array
        #ecg_segments = ecg_preprocessing(ecg_segments, 'sym8', 8, 3)
        ecg_segments = sig.decimate(ecg_segments, n = 60, q = 7, ftype = 'fir', axis = 2, zero_phase = True)
        ecg_labels = np.expand_dims(data_label, axis = 0).astype(int) # Rank-1 array
        ecg_seq_length = np.expand_dims(data.shape[1], axis = 0).astype(int) # Rank-1 array
        np.place(ecg_seq_length, ecg_seq_length > max_length, max_length)
    #
    if num_of_peaks == 5 or num_of_peaks == 6:
        #
        p = 2
        start = int(peaks[p] - peak_to_peak[p - 1] - round(0.9 * peak_to_peak[p - 2]))
        end = int(peaks[p] + peak_to_peak[p] + peak_to_peak[p + 1] + round(0.9 * peak_to_peak[p + 2]))
        #
        segment = data[:,  start : end + 1]
        #segment = ecg_preprocessing(segment, 'sym8', 8, 3)
        segment = sig.decimate(segment, n = 60, q = 7, ftype = 'fir', axis = 1, zero_phase = True)
        ecg_segments = np.expand_dims(set_to_desired_length(segment, max_len = max_length), axis = 0).astype(int) # Rank-3 array
        #
        ecg_seq_length = np.expand_dims(segment.shape[1], axis = 0) # Rank-1 array
        np.place(ecg_seq_length, ecg_seq_length > max_length, max_length)
        #
        if data_label == 2 or data_label == 6 or data_label == 7:
            #
            feature_vector = extract_features(peaks[p - 1 : p + 3])
            #
            if np.amin(feature_vector) / np.amax(feature_vector) >= .80:
                data_label = 1
        ecg_labels = np.expand_dims(data_label, axis = 0).astype(int) # Rank-1 array 
    #
    if num_of_peaks >= 7:
        # initialize
        #no_of_segments = max(1, num_of_peaks - 3)
        no_of_segments = max(1, num_of_peaks - 7)
        ecg_segments = np.zeros([no_of_segments, 12, max_length])
        ecg_labels = np.zeros([no_of_segments])
        ecg_seq_length = np.zeros([no_of_segments])
        #
        for p in np.arange(3, max(num_of_peaks - 4, 4)):
            start = int(peaks[p] - peak_to_peak[p - 1] - round(0.9 * peak_to_peak[p - 2]))
            end = int(peaks[p] + peak_to_peak[p] + peak_to_peak[p + 1] + round(0.9 * peak_to_peak[p + 2]))
            segment = data[:,  start : end + 1]
            #segment = ecg_preprocessing(segment, 'sym8', 8, 3)
            segment = sig.decimate(segment, n = 60, q = 7, ftype = 'fir', axis = 1, zero_phase = True)
            seq_length = segment.shape[1]
            segment = set_to_desired_length(segment, max_len = max_length)
            ecg_segments[p - 3, :,:] = segment
            #
            ecg_seq_length[p - 3] = seq_length
            np.place(ecg_seq_length, ecg_seq_length > max_length, max_length)
            #
            if data_label == 2 or data_label == 6 or data_label == 7:
                #
                feature_vector = extract_features(peaks[p - 1 : p + 3])
                #
                if np.amin(feature_vector) / np.amax(feature_vector) >= .80:
                    ecg_labels[p - 3] = 1
                else:
                    ecg_labels[p - 3] = data_label
            else:
                    ecg_labels[p - 3] = data_label
            
    #
    return ecg_segments, ecg_labels.astype(int) - 1, ecg_seq_length.astype(int)
#
def extract_ecg_segments_v2(peaks, file_name, data_label = None, max_length = 1000):
    #
    # read data as float32, labels as int32, sequence length as int32
    data = sio.loadmat(file_name)['ECG'][0][0][2]
    data = data.astype(np.float32)
    data_org = data
    # clean the data first
    data = ecg_preprocessing(data, 'sym8', 8, 3)
    #
    num_of_peaks = len(peaks)
    no_channels = 12
    data_length = np.expand_dims(data.shape[1], axis = 0)
    #
    peak_to_peak = np.concatenate([np.expand_dims(peaks[0], axis = 0), np.diff(peaks), data_length - peaks[-1]])
    #
    if num_of_peaks <= 4:
        data = (data - np.expand_dims(np.mean(data, axis = 1), axis = 1)) / (np.expand_dims(np.std(data, axis = 1), axis = 1) + EPS)
        data = sig.decimate(data, n = 60, q = 7, ftype = 'fir', axis = 1, zero_phase = True)
        ecg_segments = np.expand_dims(set_to_desired_length(data, max_len = max_length), axis = 0) # Rank-3 array
        #ecg_segments = ecg_preprocessing(ecg_segments, 'sym8', 8, 3)
        if data_label is not None:
        	ecg_labels = np.expand_dims(data_label, axis = 0).astype(int) # Rank-1 array
        else:
        	ecg_labels = None
        ecg_seq_length = np.expand_dims(data.shape[1], axis = 0).astype(int) # Rank-1 array
        np.place(ecg_seq_length, ecg_seq_length > max_length, max_length)
    #
    if num_of_peaks > 4:
        # initialize
        no_of_segments = max(1, num_of_peaks - 3)
        #no_of_segments = max(1, num_of_peaks - 7)
        ecg_segments = np.zeros([no_of_segments, 12, max_length])
        if data_label is not None:
        	ecg_labels = np.zeros([no_of_segments])
        else:
        	ecg_labels = None
        ecg_seq_length = np.zeros([no_of_segments])
        #
        for p in np.arange(1, num_of_peaks - 2):
            start = int(peaks[p] - peak_to_peak[p] - round(0.9 * peak_to_peak[p - 1]))
            end = int(peaks[p] + peak_to_peak[p] + peak_to_peak[p + 1] + round(0.9 * peak_to_peak[p + 2]))
            segment = data[:,  start : end + 1]
            segment = (segment - np.expand_dims(np.mean(segment, axis = 1), axis = 1)) / (np.expand_dims(np.std(segment, axis = 1), axis = 1) + EPS)
            #segment = ecg_preprocessing(segment, 'sym8', 8, 3)
            segment = sig.decimate(segment, n = 60, q = 7, ftype = 'fir', axis = 1, zero_phase = True)
            seq_length = segment.shape[1]
            segment = set_to_desired_length(segment, max_len = max_length)
            ecg_segments[p - 1, :,:] = segment
            #
            ecg_seq_length[p - 1] = seq_length
            np.place(ecg_seq_length, ecg_seq_length > max_length, max_length)
            #
            if ecg_labels is not None:
            	if data_label == 2 or data_label == 6 or data_label == 7:
            		#
            		feature_vector = extract_features(peaks[p - 1 : p + 3])
            		#
            		if np.amin(feature_vector) / np.amax(feature_vector) >= .80:
            			ecg_labels[p - 1] = 1
            		else:
            			ecg_labels[p - 1] = data_label
            	else:
                    ecg_labels[p - 1] = data_label
        #
        if data_label == 7:
        	#print('here')
        	_, PVC_segments = special_PVC(peaks, data_org)
        	#print(PVC_segments)
        	if len(PVC_segments) < no_of_segments:
        		#print('here')
        		ecg_labels[PVC_segments] = data_label
			#	ecg_labels[PVC_segments] = data_label   
    # reshape
    ecg_segments = np.transpose(ecg_segments, axes = (0, 2, 1))
    if ecg_labels is not None:
    	ecg_labels = ecg_labels.astype(int) - 1
    return ecg_segments, ecg_labels, ecg_seq_length.astype(int)
#
def special_PVC(peaks, data):
    #
    data_length = np.expand_dims(data.shape[-1], axis = 0)
    p2p = np.concatenate([np.expand_dims(peaks[0], axis = 0), np.diff(peaks), data_length - peaks[-1]])
    corr_big = np.zeros([peaks.shape[-1], data.shape[-1]])
    #
    for peak_loc, peak in enumerate(peaks, 1) :
        start = int(peak - round(0.9 * p2p[peak_loc - 1]))
        end = int(peak + round(0.9 * p2p[peak_loc]))
        template = data[:,  start : end + 1]
        #
        leads = np.arange(12)
        corr = np.zeros_like([data.shape[-1]])
        #
        for lead in leads:
            corr = corr + sig.correlate(data[lead, :], template[lead,:], mode='same')
        #
        corr_big[peak_loc - 1, :] = np.abs(corr)
    #
    corr_big = 1 / np.exp(0.00001 * ((corr_big - np.amax(corr_big)) ** 2))
    #
    candidate_peaks = np.unique(np.where(corr_big > 0.95)[0])
    #
    P = len(peaks)
    K = P - 3
    PVC_segments = np.concatenate([np.arange(start = max(0, k - 3), stop = min(k, K - 1) + 1, dtype = np.int32) for k in candidate_peaks])
    PVC_segments = np.unique(PVC_segments)
    #
    return corr_big, PVC_segments
#
def sample_batch(data_file_name, annotation_file_name, mode = 'evaluation'):
	#
	df = pd.read_csv(annotation_file_name, delimiter = ',')
	#
	RECORDS = pd.Series.as_matrix(df.Recording)
	LABELS = pd.Series.as_matrix(df.First_label)
	#
	record = re.search('A[0-9]+', data_file_name).group(0)
	if mode == 'training':
		label = LABELS[np.squeeze(np.where(RECORDS == record))]
	else:
		label = None
	#
	peaks, _ = peak_detector_with_refinement(data_file_name)
	segments, labels, lengths = extract_ecg_segments_v2(peaks, data_file_name, label)
	#
	return segments, labels, lengths
#
def process_data(file_name):
    #
    peaks, _ = peak_detector_with_refinement(file_name)
    segs, _, lens = extract_ecg_segments_v2(peaks, file_name)
    #
    return segs, lens