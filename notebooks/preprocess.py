import numpy as np
import math
import scipy.stats as stats
from scipy.io import wavfile
import time

"""Simulating Functions"""
def generate_phonemes(max_len, vocab_size):
    phoneme_n = np.random.randint(max_len)
    phonemes = np.random.randint(vocab_size,size=phoneme_n)
    zeros = np.zeros((max_len-phoneme_n), dtype=np.int)
    phonemes = np.concatenate((phonemes,zeros))
    return phonemes, phoneme_n


def generate_durations(max_len,phoneme_n, d_lower, d_upper, d_mu, d_sigma):
    duration_per_phoneme = stats.truncnorm.rvs(
              (d_lower-d_mu)/d_sigma,(d_upper-d_mu)/d_sigma,loc=d_mu,scale=d_sigma,size=phoneme_n)
    zeros = np.zeros((max_len-phoneme_n))
    duration_per_phoneme = np.concatenate((duration_per_phoneme,zeros))
    return duration_per_phoneme


def create_duration_sentence(max_len,input_vocab_size,d_lower, d_upper, d_mu, d_sigma, asn_upper, asn_lower, num_buckets, batch_size):
    # speaker id = 1
    phonemes, phoneme_n = generate_phonemes(max_len,input_vocab_size)
    duration_per_phoneme = generate_durations(max_len,phoneme_n, d_lower, d_upper, d_mu, d_sigma)
    duration_per_phoneme = assign_bucket([duration_per_phoneme]*batch_size, asn_upper, asn_lower, num_buckets)
    sentence_dict = {'phonemes':[phonemes]*batch_size,
                     'phonemes_seq_len': [phoneme_n]*batch_size,
                     'speaker_ids': 1 * np.ones((batch_size)), 
                     'durations': duration_per_phoneme
                     }
    return sentence_dict


def create_target_freqs(max_len,input_vocab_size,mu,sigma,lower,upper,batch_size):
    frame_n = np.random.randint(max_len)
    freqs = stats.truncnorm.rvs(
              (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=frame_n)
    zeros = np.zeros((max_len-frame_n))
    target_freqs = [int(frame) for frame in np.concatenate((freqs,zeros))]
    return target_freqs, frame_n

def create_frequency_sentence(max_len,input_vocab_size,mu,sigma,lower,upper,vocab_size,voiced_thresh,batch_size):
    target_freqs, frame_n = create_target_freqs(max_len,input_vocab_size,mu,sigma,lower,upper,batch_size)
    target_voiced = tag_target_voiced(target_freqs,voiced_thresh)    
    phonemes = [int(abs(freq)/upper*vocab_size) for freq in target_freqs]
    sentence_dict = {'phonemes':[phonemes]*batch_size,
                'phonemes_seq_len': [frame_n]*batch_size,
                'speaker_ids': 1 * np.ones((batch_size)), 
                'voiced_target': [target_voiced]*batch_size, 
                'frequency_target': [target_freqs]*batch_size }    
    return sentence_dict
    

"""Preprocessing Functions"""

def assign_bucket(durations, asn_upper, asn_lower, num_buckets):
    # Assign durations into buckets. Duration shape should be (batch_n, sentence_len).     
    inc = (asn_upper - asn_lower)/(num_buckets-2)
    def assign(duration):
        log_duration = np.log(duration)
        if log_duration < asn_lower:
            return 0
        elif log_duration > asn_upper:
            return num_buckets
        else:
            return int(math.ceil((log_duration-asn_lower)/inc))
    bucket_durations = [[assign(d) for d in sentence] for sentence in durations]
    return bucket_durations

def get_durations(bucket_durations,asn_lower,inc):
    # Calculate the duration by the bucket
    def get(bucket):
        log_durations = asn_lower + bucket * inc
        return np.e**log_durations
    durations = [[get(d) for d in sentence] for sentence in bucket_durations]
    return durations


def tag_target_voiced(target_freqs,voiced_thresh):
    # Check whether each frame is voiced
    # Arg durations should be real time durations
    voiced = [1 if abs(frame)>voiced_thresh else 0 for frame in target_freqs]    
    return voiced

def tag_frame_phoneme(target_freqs,seg_info):
    # Assume seg_info = [[phoneme1, phoneme2, bound_pos],...,]
    pass

def phonemes_to_frames(phonemes,bucket_durations,samp_freq,asn_lower,inc):
    # Upsample the phonemes to frames
    # This function is only used when prediction, because
    # frames_per_phoneme is given when trianing, and is infered during prediction
    durations = get_durations(bucket_durations,asn_lower,inc)
    frames_per_phoneme = [[samp_freq*duration for duration in s] for s in phonemes]
    frames = []
    zipped = np.dstack((phonemes,frames_per_phoneme))
    for sentence in zipped:
        for (phoneme,frames_n) in sentence:
            frames.extend(np.ones(frames_n,dtype=np.int)*phoneme)
    #frames = [np.ones(frames_n,dtype=np.int)*phoneme for (phoneme,frames_n) in sentence] for sentence in zipped]
    return frames



