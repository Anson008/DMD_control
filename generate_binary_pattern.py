from PatternSequenceGenerator import PatternSequenceGenerator
from PeriodGenerator import PeriodGenerator
import matplotlib.pyplot as plt
import time

# Generate a list of prime numbers
upper = 542  # Upper limit to search for prime numbers
periods = PeriodGenerator.prime_number_list(upper)
# periods = [3]

start = time.time()
patt = PatternSequenceGenerator(14100, 10, 10, 20)
patt.generate_binary_patterns(periods, mode='nonuniform')
patt.pad()
end = time.time()
print(f"It takes {end - start:.2f}s to generate patterns.")

# Get time series along z axis at position (y, x)
"""ts00 = patt.get_time_series(x=9, y=9)
plt.plot(ts00, 'r.-')
plt.show()"""

file_name = 'binary_14100Frames_10X10_scale20_pad1'

start = time.time()
patt.save_to_images(directory="./" + file_name, prefix=file_name, bit_level=1)
end = time.time()
print(f"It takes {end - start:.2f}s to save patterns to files.")