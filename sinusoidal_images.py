from PatternSequenceGenerator import PatternSequenceGenerator
import numpy as np


# Pattern parameters
width = 2
height = 2
scale = 300
beta = 100
frames = width * height * beta

pad = 0
decompose = 1

patt = PatternSequenceGenerator(frames, width, height, scale)
pattern = patt.generate_sinusoidal_patterns()

# Get time series along z axis at position (y, x)
"""ts00 = patt.get_time_series(x=9, y=9)
plt.plot(ts00, 'r.-')
plt.show()"""

path = "E:/Data_exp/Freq_Encoded_Data/patterns/"
file_name = f'sin_{frames}Frames_{width}X{height}_scale{scale}_pad{pad}_bR{decompose}'

video_path = "E:/Data_exp/Freq_Encoded_Data/videos"
video_name = f'sin_{frames}Frames_{width}X{height}_scale{scale}_pad{pad}_bR{decompose}.avi'

if decompose == 0:
    bit_level = 8
    patt.save_to_images(pattern, directory=path + file_name, prefix=file_name, bit_level=bit_level)
    patt.save_to_video(pattern, 60, directory=video_path, filename=video_name)
else:
    bit_level = 1
    pattern = patt.gray2binary(pattern).astype(np.uint8)
    patt.save_to_images(pattern, bit_level=bit_level, directory=path + file_name, prefix=file_name)
    patt.save_to_video(pattern, 60, directory=path, filename=file_name, pattern_type='binary')


