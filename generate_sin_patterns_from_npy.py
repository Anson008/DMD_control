from PatternSequenceGenerator import PatternSequenceGenerator
import time
import os

# Pattern parameters
width = 20
height = 20
scale = 7
beta = 100
frames = width * height * beta

path_data = "E:/Data_exp/Freq_Encoded_Data/patterns/"
name_npy = "sin_40000Frames_20X20_scale1_pad0_bR0.npy"

path_pattern = "E:/Data_exp/Freq_Encoded_Data/test/"

start = time.time()
patt = PatternSequenceGenerator(frames, width, height, scale)
patt.generate_patterns_from_npy(os.path.join(path_data, name_npy), path_pattern, n_batch=10)
end = time.time()
print(f"Taken {end - start: .2f} s to complete.")
