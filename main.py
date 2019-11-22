import pyvisa
import sys
import time
import numpy as np
from dso6032 import DSO6034A

visa_rm = pyvisa.ResourceManager("C:\\Windows\\System32\\visa32.dll")
vr = visa_rm.list_resources()
print("VISA resources:", vr)

my_scope = DSO6034A(visa_address=vr[0])
my_scope.connect()
my_scope.find_channels()
x_time, y_wave = my_scope.acquisition()
my_scope.save_data(x_time, y_wave, file_name="test", directory="E:\\Xingchen Zhao\\code_xz\\debug_data\\", file_type=".csv")
