from PatternSequenceGenerator import PatternSequenceGenerator
import numpy as np
import time

# Pattern parameters
width = 60
height = 20
scale = 10
beta = 1
frames = width * height * beta
bit_level = 8
pad = 0
decompose = 1

start = time.time()
patt = PatternSequenceGenerator(frames, width, height, scale)
pattern = patt.generate_sinusoidal_patterns()
end = time.time()
print(f"It takes {end - start:.2f} s to generate patterns.")

# Get time series along z axis at position (y, x)
"""ts00 = patt.get_time_series(x=9, y=9)
plt.plot(ts00, 'r.-')
plt.show()"""

"""
path = "E:/Data_exp/Freq_Encoded_Data/patterns/"
# file_name = 'white_1920X1080'
file_name = f'sin_{frames}Frames_{width}X{height}_scale{scale}_pad{pad}_bR{decompose}'
# file_name = f'testSin_{frames}Frames_{width}X{height}_scale{scale}_pad{pad}_bR{decompose}'

start = time.time()
patt.save_to_images(pattern, directory=path + file_name, prefix=file_name, bit_level=bit_level)
end = time.time()
print(f"It takes {end - start:.2f} s to save patterns to files.")
"""


path = "E:/Data_exp/Freq_Encoded_Data/videos"
file_name = f'sin_{frames}Frames_{width}X{height}_scale{scale}_pad{pad}_bR{decompose}.avi'

if decompose == 0:
    patt.save_to_video(pattern, 60, directory=path, filename=file_name)
else:
    pattern = patt.gray2binary(pattern).astype(np.uint8)
    patt.save_to_video(pattern, 60, directory=path, filename=file_name, pattern_type='binary')



