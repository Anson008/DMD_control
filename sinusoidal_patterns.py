from PatternSequenceGenerator import PatternSequenceGenerator
import matplotlib.pyplot as plt
import time

# Pattern parameters
width = 60
height = 20
scale = 10
beta = 100
rs_factor = 10
frames = width * height * beta

decompose = 0  # 0 - gray scale; 1 - decompose to binary.
pad = 0  # 0 - no pad; 1 - pad

start = time.time()
patt = PatternSequenceGenerator(frames, width, height, scale)
pattern = patt.generate_sinusoidal_patterns(rs_factor=rs_factor)

if decompose == 1:
    bit_level = 1
    pattern = patt.gray2binary(pattern)
else:
    bit_level = 8

if pad == 1:
    pattern = patt.pad(pattern, width=1920, height=height*scale)
end = time.time()
print(f"Taken {end - start:.2f} s to generate patterns.")

# Get time series along z axis at position (y, x)
"""ts00 = patt.get_time_series(x=9, y=9)
plt.plot(ts00, 'r.-')
plt.show()"""

path = "E:/Data_exp/Freq_Encoded_Data/patterns/"
# file_name = 'white_1920X1080'
file_name = f'sin_{frames}Frames_{width}X{height}_scale{scale}_rs{rs_factor}_pad{pad}_bR{decompose}'
# file_name = f'testSin_{frames}Frames_{width}X{height}_scale{scale}_pad{pad}_bR{decompose}'

start = time.time()
if decompose == 1:
    patt.save_to_images(pattern, directory=path + file_name, prefix=file_name, bit_level=bit_level)
else:
    patt.save_to_npy(pattern, path, file_name)
end = time.time()
print(f"Taken {end - start:.2f} s to save patterns.")
