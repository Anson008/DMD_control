from PatternSequenceGenerator import PatternSequenceGenerator
from PeriodGenerator import PeriodGenerator
import matplotlib.pyplot as plt
import time

patt = PatternSequenceGenerator(1, 10, 10, 15)
cali = patt.make_calibration_pattern()
cali_padded = patt.pad(cali)

file_name = 'calib_02.png'
path = "E:/Data_exp/Freq_Encoded_Data/patterns/calibration"
patt.save_single_pattern(cali_padded, path, file_name)