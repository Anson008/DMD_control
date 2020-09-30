from PatternSequenceGenerator import PatternSequenceGenerator
from PeriodGenerator import PeriodGenerator
import matplotlib.pyplot as plt
import time

# Generate a list of prime numbers
# upper = 8  # Upper limit to search for prime numbers
# periods = PeriodGenerator.prime_number_list(upper)
# periods = PeriodGenerator.prime_numbers()
periods = [i for i in range(2, 514, 2)]
#periods = [2]

start = time.time()
patt = PatternSequenceGenerator(15000, 10, 10, 15)
pattern = patt.generate_binary_patterns(periods, mode='nonuniform')
pattern_padded = patt.pad(pattern)
end = time.time()
print(f"It takes {end - start:.2f}s to generate patterns.")

# Get time series along z axis at position (y, x)
"""ts00 = patt.get_time_series(x=9, y=9)
plt.plot(ts00, 'r.-')
plt.show()"""

file_name = 'binary_15kFrames_10X10_scale15_pad1_natural'
#file_name = 'testB_150Frames_16X16_scale16_pad1_natural'

start = time.time()
patt.save_to_images(pattern_padded, directory="E:/Data_exp/Freq_Encoded_Data/patterns/" + file_name,
                    prefix=file_name, bit_level=1)
end = time.time()
print(f"It takes {end - start:.2f}s to save patterns to files.")
