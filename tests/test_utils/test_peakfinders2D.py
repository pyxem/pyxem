import pytest
import numpy as np
from pyxem.utils.peakfinders2D import *

@pytest.fixture
def single_peak():
    pattern = np.zeros((128,128))
    pattern[40:43,40:43] = 1 #index 40,41,42 are greater than zero
    return pattern

@pytest.fixture
def double_peak():
    pattern = np.zeros((128,128))
    pattern[40:43,40:43] = 1 #index 40,41,42 are greater than zero
    pattern[70,21]    = 1 #index 70 and 21 are greater than zero   
    return pattern



@pytest.mark.skip(reason="This method should probably be removed")
def test_fp_minmax(single_peak):
    peaks = find_peaks_minmax(single_peak,threshold=0.5)

@pytest.mark.skip(reason="This method should probably be removed")
def test_fp_regp(single_peak):
    peaks = find_peaks_regionprops(single_peak)
    assert peaks[0,0] > 39.5 
    assert peaks[0,0] < 42.5
    assert peaks[0,0] == peaks[0,1]

@pytest.mark.skip(reason="This method can also probably go")
def test_fp_max(single_peak):
    peaks = find_peaks_max(single_peak,alpha=0.00001)
    assert peaks[0,0] > 39.5 
    assert peaks[0,0] < 42.5
    assert peaks[0,0] == peaks[0,1]
    ## Fails at double peaks

def test_fp_dog(single_peak,double_peak):
    peaks = find_peaks_dog(single_peak)
    assert peaks[0,0] > 39.5 
    assert peaks[0,0] < 42.5
    assert peaks[0,0] == peaks[0,1]
    peaks = find_peaks_dog(double_peak)
    assert (np.array([71,21]) in peaks) 

def test_fp_log(single_peak,double_peak):
    peaks = find_peaks_log(single_peak)
    assert peaks[0,0] > 39.5 
    assert peaks[0,0] < 42.5
    assert peaks[0,0] == peaks[0,1]
    peaks = find_peaks_log(double_peak)
    assert (np.array([71,21]) in peaks) 

def test_fp_zaef(single_peak,double_peak):
    peaks = find_peaks_zaefferer(single_peak)
    assert peaks[0,0] > 39.5 
    assert peaks[0,0] < 42.5
    assert peaks[0,0] == peaks[0,1]
    peaks = find_peaks_zaefferer(double_peak)
    assert (np.array([71,21]) in peaks) 
    
def test_fp_stat(single_peak,double_peak):
    peaks = find_peaks_stat(single_peak)
    assert peaks[0,0] > 39.5 
    assert peaks[0,0] < 42.5
    assert peaks[0,0] == peaks[0,1]
    peaks = find_peaks_stat(double_peak)
    assert (np.array([71,21]) in peaks) 