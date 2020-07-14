from PatternSequenceGenerator import PatternSequenceGenerator
import matplotlib.pyplot as plt

periods = [500]
patt = PatternSequenceGenerator(10000, 2, 2, 100)
patt.generate_binary_patterns(periods)
patt.pad()

# Get time series along z axis at position (y, x)
"""ts00 = patt.get_time_series(x=0, y=0)
plt.plot(ts00)
plt.show()"""

file_name = 'testB_10kFrames_2X2_scale100_T500_pad1'
patt.save_to_images(directory="./" + file_name, prefix=file_name, bit_level=1)
