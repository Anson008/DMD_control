from PatternSequenceGenerator import PatternSequenceGenerator
import matplotlib.pyplot as plt
import time

# Pattern parameters
frames = 1000
width = 10
height = 10
scale = 20
decompose = 0  # 0 - gray scale; 1 - decompose to binary.

start = time.time()
patt = PatternSequenceGenerator(frames, width, height, scale)
pattern = patt.generate_sinusoidal_patterns()
if decompose == 1:
    bit_level = 1
    pattern = patt.gray2binary(pattern)
else:
    bit_level = 8
pattern = patt.pad(pattern, width=1920, height=height*scale)
end = time.time()
print(f"It takes {end - start:.2f} s to generate patterns.")

# Get time series along z axis at position (y, x)
"""ts00 = patt.get_time_series(x=9, y=9)
plt.plot(ts00, 'r.-')
plt.show()"""

# file_name = 'white_1920X1080'
# file_name = f'sin_{frames}Frames_{width}X{height}_scale{scale}_pad1_bR1'
file_name = f'testSin_{frames}Frames_{width}X{height}_scale{scale}_pad1_bR{decompose}'

start = time.time()
patt.save_to_images(pattern, directory="E:/Data_exp/Freq_Encoded_Data/patterns/" + file_name,
                    prefix=file_name, bit_level=bit_level)
end = time.time()
print(f"It takes {end - start:.2f} s to save patterns to files.")