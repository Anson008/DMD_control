from PatternSequenceGenerator import PatternSequenceGenerator

frames = 1
width = 192
height = 108
patt = PatternSequenceGenerator(frames, width, height, scale=10)
pattern = patt.generate_random_pattern()
patt.save_single_pattern(pattern, directory='E:\\Data_exp\\Freq_Encoded_Data\\patterns', filename='random_test.png')