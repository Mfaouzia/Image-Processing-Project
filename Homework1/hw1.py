"""
姓名：齐琪格
学号：6319000163
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from PIL import Image


def judge(img1, img2, ratio):
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)

    diff = np.abs(img1 - img2)
    count = np.sum(diff > 1)

    assert count == 0, f'ratio={ratio}, Error!'
    print(f'ratio={ratio}, Success!')


def get_gt(img, ratio):
    new_h = int(img.shape[0] * ratio)
    new_w = int(img.shape[1] * ratio)
    gt = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return gt


# to do
def resize(img, ratio):
    """
    禁止使用cv2、torchvision等视觉库
    type img: ndarray(uint8)
    type ratio: float
    rtype: ndarray(uint8)
    """
    src_height = img.shape[0]
    src_width = img.shape[1]
    dst_height = int(src_height * ratio)
    dst_width = int(src_width * ratio)
    # print("src_width, src_height, dst_width, dst_height:", src_width, src_height, dst_width, dst_height)
    resized = np.zeros((dst_height, dst_width, 3), np.uint8)

    scale_x = float(src_width) / dst_width
    scale_y = float(src_height) / dst_height

    for dst_x in range(dst_width):
        src_x = (dst_x + 0.5) * scale_x - 0.5
        src_x0 = math.floor(src_x)
        A = src_x - src_x0

        if src_x < 0:
            src_x0 = 0
            A = 0
        if src_x >= src_width - 1:
            src_x0 = src_width - 2
            A = 1

        for dst_y in range(dst_height):
            src_y = (dst_y + 0.5) * scale_y - 0.5
            src_y0 = math.floor(src_y)
            B = src_y - src_y0

            if src_y < 0:
                src_y0 = 0
                B = 0
            if src_y >= src_height - 1:
                src_y0 = src_height - 2
                B = 1

            # print("dst_x, dst_y:", dst_x, dst_y)
            # print("x0,x1,y0,y1:", src_x0, src_x1, src_y0, src_y1)
            for k in range(3):
                value0 = (1 - A) * img[src_y0, src_x0, k] + A * img[src_y0, src_x0 + 1, k]
                value1 = (1 - A) * img[src_y0 + 1, src_x0, k] + A * img[src_y0 + 1, src_x0 + 1, k]
                resized[dst_y, dst_x, k] = int((1 - B) * value0 + B * value1)
    return resized


def show_images(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':
    ratios = [0.5, 0.8, 1.2, 1.5]

    img = cv2.imread('E:/pythonProject/hw1/images/img_1.jpeg')  # type(img) = ndarray, 一共有三张图片，都可以尝试
    start_time = time.time()
    for ratio in ratios:
        gt = get_gt(img, ratio)
        resized_img = resize(img, ratio)
        if ratio == 0.8:
        show_images(gt, resized_img)  # 这里加就行
        judge(gt, resized_img, ratio)
    end_time = time.time()
    total_time = end_time - start_time

    print(f'用时{total_time:.4f}秒')
    print('Pass')
