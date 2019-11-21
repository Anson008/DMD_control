import numpy as np

a1 = np.zeros((3, 4))
a2 = np.arange(0, 12).reshape((3, 4))
print(a1[0, :])
print(a2[0, :])
c = 3.1415
d = 1
print("{:d}".format(d))

print("""\n\nIt took {:.2f} seconds to transfer and scale {:d} channel(s). 
Each channel had {:d} points.\n""".format(c, d, d))
