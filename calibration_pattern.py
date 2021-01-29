from PatternSequenceGenerator import PatternSequenceGenerator
from PeriodGenerator import PeriodGenerator
import matplotlib.pyplot as plt
import time

patt = PatternSequenceGenerator(1, 1000, 1000, 1)
cali = patt.make_calibration_pattern()
cali_padded = patt.pad(cali)

file_name = 'calib_1000by1000_02.png'
path = "E:/Data_exp/Freq_Encoded_Data/patterns/calibration"
patt.save_single_pattern(cali_padded, path, file_name)