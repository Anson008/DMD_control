from PatternSequenceGenerator import PatternSequenceGenerator
import time
import os

# Pattern parameters
width = 60
height = 20
scale = 10
beta = 100
frames = width * height * beta

path_data = "E:/Data_exp/Freq_Encoded_Data/patterns/"
name_npy = "sin_120000Frames_60X20_scale1_pad0_bR0.npy"

# path_pattern = "E:/Data_exp/Freq_Encoded_Data/test/"
path_pattern = "E:/Data_exp/Freq_Encoded_Data/patterns/"

start = time.time()
patt = PatternSequenceGenerator(frames, width, height, scale)
patt.generate_patterns_from_npy(os.path.join(path_data, name_npy), path_pattern, n_batch=100)
end = time.time()
print(f"Taken {end - start:.2f} s to complete.")
