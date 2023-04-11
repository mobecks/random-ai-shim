import numpy as np
import matplotlib.pyplot as plt

# returns the desired standard deviation to construct a normal distribution 
# for a predefined SNR, given an input array
def get_noise_avg(pred_numpy, snr = 1):
    power = pred_numpy **2
    avg = np.mean(power)
    db = 10 * np.log10(avg)
    noise_avg_db = db - snr
    noise_avg = 10 **(noise_avg_db / 10)
    return np.sqrt(noise_avg)
  
    
# %% Scores and Statistics #################################################################################################

# Return the SNR of a spectrum (as in Magritek)
# Signal is the maximum peak.
# Noise is the rms of the last 1/4 of the spectrum.  
def getSNR(array):   
    ry = array.real
    sig = max(ry)
    ns = ry[int(3*len(array)/4):-1]
    return sig/np.std(ns)

def gaussian(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def quartile_skewness(array):
    p75 = np.percentile(array, 75)
    p25 = np.percentile(array, 25)
    median = np.percentile(array, 50)
    return ( ( (p75 - median) - (median - p25) ) / (p75 - p25) )   # 0 for symmetric

def g(arr):             # skewness
    m = arr.mean()
    s = arr.std()
    return 1/len(arr) * np.sum( ( (arr - m)/s )**3)
    
# R^2 score
def r2_score(pred,gt):
    mean = np.mean(gt)
    SStot = np.sum( (gt - mean)**2 )
    SSres = np.sum( (gt - pred)**2 )
    return 1 - SSres/SStot