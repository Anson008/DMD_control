import numpy as np
import time
import cv2
import os
from periods import PeriodGenerator


class BinaryImage(object):
    def __init__(self, frames, width, height, xy_scale, padding, periods):
        """
        :_frames: int, number of frames in the image sequence
        :_width: int, image width
        :_height: int, image height
        :_periods: list of int, the periods of signals along z axis specifying by (y, x)
        :_width: tuple, ((before1, after1), (before2, after2)) specifies the number of values padded to the edges of each axis.
        :_pixels: numpy array, store the pixel values of the final image
        """
        self._frames = frames
        self._width = width
        self._height = height
        self._xy_scale = xy_scale
        self._padding = padding
        self._periods = periods
        self._pixels = None

    def get_img_shape(self):
        """
        :return: tuple, the shape of image sequence in (frames, height, width).
        """
        return self._frames, self._height, self._width, self._xy_scale

    def get_pixels(self):
        """
        :return: numpy array, pixel values of the image sequence
        """
        return self._pixels

    def save_to_npy(self, directory='./raw_data', filename='test'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = directory + '/' + filename + '.npy'
        np.save(file_path, self._pixels)

    def generate_images(self):
        """
        Generate binary images according to the given periods.
        """
        self._pixels = np.zeros((self._frames, self._height * self._xy_scale, self._width * self._xy_scale), dtype=np.uint8)
        for z in range(self._frames):
            for y in range(self._height * self._xy_scale):
                for x in range(self._width * self._xy_scale):
                    # first define the mapping between the index of period value and (y, x)
                    i_period = (x // self._xy_scale) + self._width * (y // self._xy_scale)
                    if (z // self._periods[i_period]) % 2 == 0:
                        self._pixels[z][y][x] = 255
                    else:
                        self._pixels[z][y][x] = 0

    def generate_sinusoidal_images(self, time=2000, freq_multiplier=0.1):
        self._pixels = np.zeros((self._frames, self._height * self._xy_scale, self._width * self._xy_scale), dtype=np.uint8)
        for y in range(self._height * self._xy_scale):
            for x in range(self._width * self._xy_scale):
                # first define the mapping between the index of period value and (y, x)
                # i_period = (x // self._xy_scale) + self._width * (y // self._xy_scale) + 1
                # self._pixels[:, y, x] = 128 * np.sin(2 * np.pi / self._periods[i_period] * np.linspace(0, time, num=self._frames)) + 127
                freq = ((x // self._xy_scale) + self._width * (y // self._xy_scale) + 1) * freq_multiplier
                self._pixels[:, y, x] = 128 * np.sin(2 * np.pi * freq *
                                                     np.linspace(0, time, num=self._frames)) + 127

    def print_through_time(self, xy_range):
        """
        :xy_range: list of tuple, specify the (x, y) range of image we want to print out along z axis.
                   [(x1, x2), (y1, y2)]
        """
        print("-"*10, "Preview of signal along z axis", "-"*10)
        for y in range(xy_range[1][0], xy_range[1][1]):
            for x in range(xy_range[0][0], xy_range[0][1]):
                print(self._pixels[:, y, x])
        print("-"*10, "End of printing", "-"*10, "\n")

    def preview(self):
        for z in range(self._frames):
            cv2.imshow('Preview of images', self._pixels[z, :, :])
            if cv2.waitKey(0) == ord('q'):
                break
        cv2.destroyAllWindows()

    def add_padding(self):
        self._pixels = np.pad(self._pixels, ((0, 0), (self._padding[0][0], self._padding[0][1]), (self._padding[1][0], self._padding[1][1])),
                              'constant', constant_values=((0, 0),))

    def undo_padding(self):
        self._pixels = self._pixels[:, self._padding[0][0]:-self._padding[0][1], self._padding[1][0]:-self._padding[1][1]]

    def make_calibration_image(self):
        w, h = self._width * self._xy_scale, self._height * self._xy_scale
        img = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                c1i = (int(0.1*h) <= i <= int(0.3*h) or
                       int(0.4*h) <= i <= int(0.6*h) or
                       int(0.7*h) <= i <= int(0.9*h))
                c1j = (int(0.4*w) <= j <= int(0.6*w))
                img[i, j] = 255 if c1i or c1j else 0
        img = np.pad(img, ((self._padding[0][0], self._padding[0][1]), (self._padding[1][0], self._padding[1][1])),
                     'constant', constant_values=((0, 0),))
        return img

    @staticmethod
    def save_image(img, directory='./calibration', filename='calib_01', fmt='.png'):
        """
        Save a single image to a user specified directory.
        :para directory: string, destination directory.
        :para filename: string, file name without extension.
        :para fmt: string, format of the image files. Default is '.png'.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = directory + "/" + filename + fmt
        cv2.imwrite(file_path, img)

    def make_image_sequence(self, directory='./img_sequence', filename='test', fmt='.png'):
        """
        Save image sequence to a user specified directory, with auto naming from "0" to "# of images - 1".
        :para directory: string, destination directory.
        :para fmt: string, format of the image files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        for z in range(self._frames):
            file_path = directory + "/" + filename + '_' + "{:d}".format(z).zfill(len(str(z))) + fmt
            cv2.imwrite(file_path, self._pixels[z, :, :])

    def make_binary_video(self, fps, directory='./videos', filename='video1', fmt='.avi', threshold=127):
        """
        Save image sequence as a video to user specified directory.
        :para fps: int, frame rate of the video.
        :para directory: string, destination directory.
        :para filename: string, file name of the video, without extension.
        :para fmt: string, format of the image files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        fs, w, h = self._pixels.shape
        file_path = directory + '/' + filename + fmt
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(file_path, fourcc, fps, (w, h), False)
        for z in range(fs):
            # Convert gray scale to binary.
            thresh, frame_b = cv2.threshold(self._pixels[z, :, :], threshold, 255, cv2.THRESH_BINARY)
            video.write(frame_b)
        video.release()

    def make_grayscale_video(self, fps, directory='./videos', filename='video1', fmt='.avi'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        fs, w, h = self._pixels.shape
        file_path = directory + '/' + filename + fmt
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(file_path, fourcc, fps, (w, h), False)
        for z in range(fs):
            video.write(self._pixels[z, :, :])
        video.release()


if __name__ == "__main__":
    # Load the leading 100 prime numbers
    periods = PeriodGenerator()
    prime_num = periods.prime_numbers()

    # Set image padding, create an object of BinaryImage class.
    padding = ((20, 20), (20, 20))
    binary_img = BinaryImage(20000, 4, 4, 5, padding, prime_num)

    # Get image shape.
    frames, height, width, scale = binary_img.get_img_shape()

    # Generate image sequence without padding.
    # start = time.time()
    # binary_img.generate_images()
    """print("\nIt took {:.2f} s to generate image sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual images should be ({:d}, {:d}, {:d}), without taking account into padding."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))"""

    # Generate sinusoidal image sequence
    start = time.time()
    binary_img.generate_sinusoidal_images(time=2000)
    print("\nIt took {:.2f} s to generate image sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual images should be ({:d}, {:d}, {:d}), without taking account into padding."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))

    # Preview image pixel values along time axis.
    # binary_img.print_through_time([(0, 2), (0, 2)])

    # Add padding to images.
    binary_img.add_padding()

    # Undo padding to images if necessary.
    # binary_img.undo_padding(pad_width=padding)

    # Preview image sequence frame by frame if necessary.
    # binary_img.preview()

    # Get pixel values if necessary.
    # image = binary_img.get_pixels()
    # print("\nShape of generated images:", image.shape)

    # Save pixel values to .npy file if necessary.
    binary_img.save_to_npy(filename='testSin_20kFrames_4X4')

    # Make calibration images.
    # calib_img = binary_img.make_calibration_image()
    # binary_img.save_image(calib_img, filename='calib_8by8')

    # binary_img.make_image_sequence()
    # binary_img.make_binary_video(fps=20, filename='d')

    # Make gray scale video
    # binary_img.make_grayscale_video(fps=20, filename='sinusoidal_200kFrames_20fps_8X8_natNumFreq')




