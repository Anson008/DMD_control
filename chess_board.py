from PatternSequenceGenerator import PatternSequenceGenerator

frames = 1
n = 8
patt = PatternSequenceGenerator(frames, n, n, scale=150)
pattern = patt.make_chess_board()
patt.save_single_pattern(pattern, directory='E:\\Data_exp\\Freq_Encoded_Data\\patterns', filename='chess_board_scale150.png')