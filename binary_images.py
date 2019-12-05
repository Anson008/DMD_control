import numpy as np
import time
import cv2


class BinaryImage(object):
    def __init__(self, frames, width, height, periods):
        """
        :_frames: int, number of frames in the image sequence
        :_width: int, image width
        :_height: int, image height
        :_periods: list of int, the periods of signals along z axis specifying by (y, x)
        :_pixels: numpy array, store the pixel values of the final image
        """
        self._frames = frames
        self._width = width
        self._height = height
        self._periods = periods
        self._pixels = None

    def get_img_shape(self):
        """
        :return: tuple, the shape of image sequence in (frames, height, width).
        """
        return self._frames, self._height, self._width

    def get_pixels(self):
        return self._pixels

    def generate_images(self):
        """
        Generate binary images according to the given periods.
        """
        self._pixels = np.zeros((self._frames, self._height, self._width), dtype=np.uint8)
        for z in range(self._frames):
            for y in range(self._height):
                for x in range(self._width):
                    # first define the mapping between the index of period value and (y, x)
                    i_period = x + self._width * y
                    if (z // self._periods[i_period]) % 2 == 0:
                        self._pixels[z][y][x] = 255
                    else:
                        self._pixels[z][y][x] = 0

    def print_through_time(self, xy_range):
        """
        :*xy_range: list of tuple, specify the (x, y) range of image we want to print out along z axis.
                   [(x1, x2), (y1, y2)]
        """
        print("-"*10, "Preview of signal along z axis", "-"*10)
        for y in range(xy_range[1][0], xy_range[1][1]):
            for x in range(xy_range[0][0], xy_range[0][1]):
                print(self._pixels[:, y, x])
        print("-"*10, "End of printing", "-"*10, "\n")

    def make_binary_video(self):
        fps, w, h = 1, self._width, self._height
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter('./binary_image.avi', fourcc, fps, (w, h), False)
        for z in range(self._frames):
            thresh, frame_b = cv2.threshold(self._pixels[z, :, :], 127, 255, cv2.THRESH_BINARY)
            video.write(frame_b)
        video.release()


if __name__ == "__main__":
    # Leading 100 prime numbers
    prime_num = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
                 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
                 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
                 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
                 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]

    binary_img = BinaryImage(240, 2, 2, prime_num)
    frames, height, width = binary_img.get_img_shape()

    start = time.time()
    binary_img.generate_images()
    print("\nIt took {:.2f} s to generate image sequence of shape "
          "({:d}, {:d}, {:d}) (frames, height, width).\n".format(time.time() - start, frames, height, width))

    # binary_img.print_through_time([(0, 2), (0, 2)])

    # image = binary_img.get_pixels()
    # print("Shape of image:", image.shape)

    binary_img.make_binary_video()




