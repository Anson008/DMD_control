import numpy as np
import cv2
import os
import time


def read_images(path='E:/Data_exp/imaging_T_60fps_8X8_4min_files'):
    file_list = os.listdir(path)
    file_list.sort()
    img = cv2.imread(path + '/' + file_list[0], flags=0)

    # Add time axis
    row, column = img.shape[0], img.shape[1]
    img = img.reshape((1, row, column))

    # Read image as grayscale
    for f in file_list[1:]:
        new = cv2.imread(path + '/' + f, flags=0).reshape((1, row, column))
        img = np.concatenate((img, new), axis=0)
    return img


def save_image_to_npy(data, path="E:/code_xz/DMD_control/raw_data/",
                      filename='T_60fps_8X8_4min.npy'):
    np.save(path + filename, data)


def preview(data):
    for f in range(data.shape[0]):
        cv2.imshow('test', data[f, ...])
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_path = 'E:/Data_exp/test_img'
    start = time.time()
    images = read_images(path=test_path)
    # print("Image shape:", images.shape)
    save_image_to_npy(images, filename='test_imgs.npy')
    end = time.time()
    print("It took {:.2f} seconds to convert the image data to .npy format.".format(end - start))

    load_path = 'E:/code_xz/DMD_control/raw_data/test_imgs.npy'
    images_loaded = np.load(load_path)
    print(images_loaded.shape)
    print(images.dtype)

