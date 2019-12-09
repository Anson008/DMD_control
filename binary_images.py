import numpy as np
import time
import cv2
import os
from periods import PeriodGenerator


class BinaryImage(object):
    def __init__(self, frames, width, height, xy_scale, periods):
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
        self._xy_scale = xy_scale
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

    def preview(self):
        for z in range(self._frames):
            cv2.imshow('Preview of images', self._pixels[z, :, :])
            if cv2.waitKey(0) == ord('q'):
                break
        cv2.destroyAllWindows()

    def add_padding(self, pad_width):
        """
        :para width: tuple, ((before1, after1), (before2, after2)) specifies the number of values padded to the edges of each axis.
        :return: numpy array, padded array.
        """
        self._pixels = np.pad(self._pixels, ((0, 0), (pad_width[0][0], pad_width[0][1]), (pad_width[1][0], pad_width[1][1])),
                              'constant', constant_values=((0, 0),))

    def undo_padding(self, pad_width):
        """
        :para width: tuple, ((before1, after1), (before2, after2)) specifies the number of values removed against the edges of each axis.
        :return: numpy array, padded array.
        """
        self._pixels = self._pixels[:, pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]

    def make_image_sequence(self, directory='./img_sequence', fmt='.png'):
        """
        Save image sequence to a user specified directory, with auto naming from "0" to "# of images - 1".
        :para directory: string, destination directory.
        :para fmt: string, format of the image files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        for z in range(self._frames):
            file_path = directory + "/" + "{:d}".format(z).zfill(len(str(z))) + fmt
            cv2.imwrite(file_path, self._pixels[z, :, :])

    def make_binary_video(self, fps, directory='./videos', filename='video1', fmt='.avi'):
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
            thresh, frame_b = cv2.threshold(self._pixels[z, :, :], 127, 255, cv2.THRESH_BINARY)
            video.write(frame_b)
        video.release()


if __name__ == "__main__":
    # Leading 100 prime numbers
    periods = PeriodGenerator()
    prime_num = periods.prime_numbers()

    padding = ((200, 200), (200, 200))
    binary_img = BinaryImage(24, 2, 2, 100, prime_num)
    frames, height, width, scale = binary_img.get_img_shape()

    start = time.time()
    binary_img.generate_images()
    print("\nIt took {:.2f} s to generate image sequence of shape "
          "({:d}, {:d}, {:d}). \nThe actual height and width are scaled by a factor of {:d}."
          "\nThe shape of actual images should be ({:d}, {:d}, {:d}), without taking account into padding."
          .format(time.time() - start, frames, height, width, scale, frames, height * scale, width * scale))

    # binary_img.print_through_time([(0, 2), (0, 2)])

    binary_img.add_padding(pad_width=padding)
    # binary_img.undo_padding(pad_width=padding)
    # binary_img.preview()
    image = binary_img.get_pixels()
    print("\nShape of generated images:", image.shape)
    # binary_img.save_to_npy()



    #binary_img.make_image_sequence()
    #binary_img.make_binary_video(fps=24, filename='24KFrames_24Fps_8by8_200Padding')




