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


def read_video(path='E:/Data_exp/image_T_8X8_60fps.avi'):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames in the video: {:d}".format(frame_count))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res = np.empty((frame_count, frame_height, frame_width), np.dtype('uint8'))

    count = 0
    ret = True
    while count < frame_count and ret:
        ret, frame = cap.read()
        res[count] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
    cap.release()
    return res


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
    # test_path = 'E:/Data_exp/test_img'
    start = time.time()
    # images = read_images(path='E:/Data_exp/60fps_reference_24fps_8X8_sample_2min_files')
    images = read_video(path='E:/Data_exp/T_bottomright_120fps_sin.avi')
    # print("Image shape:", images.shape)
    save_image_to_npy(images, filename='T_bottomright_120fps_sin.npy')
    end = time.time()
    print("It took {:.2f} seconds to convert the image data to .npy format.".format(end - start))



