import numpy as np
import matplotlib.pyplot as plt
from binary_images import BinaryImage
from periods import PeriodGenerator


class SimulationEngine:
    pass





if __name__ == "__main__":
    periods = PeriodGenerator()
    prime_num = periods.prime_numbers()
    binary_images = BinaryImage(24, 2, 2, 100)
    frame, width, height, scale = binary_images.get_img_shape()


