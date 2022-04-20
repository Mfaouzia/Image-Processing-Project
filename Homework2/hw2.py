"""
学号:6319000163
姓名:齐琪格
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def judge(src_img, spec_img, ref_img):
    src_pixel_num = src_img.shape[0] * src_img.shape[1]
    ref_pixel_num = ref_img.shape[0] * ref_img.shape[1]

    src_hist = cv2.calcHist([src_img], [0], None, [256], [0, 255]) / src_pixel_num
    spec_hist = cv2.calcHist([spec_img], [0], None, [256], [0, 255]) / src_pixel_num
    ref_hist = cv2.calcHist([ref_img], [0], None, [256], [0, 255]) / ref_pixel_num

    plt.subplot(1, 3, 1)   # debug时可打开
    plt.hist(src_img.ravel(), 256, [0, 255])
    plt.subplot(1, 3, 2)
    plt.hist(ref_img.ravel(), 256, [0, 255])
    plt.subplot(1, 3, 3)
    plt.hist(spec_img.ravel(), 256, [0, 255])
    plt.show()

    cur_loss = np.sum(np.abs(spec_hist - ref_hist))
    pre_loss = np.sum(np.abs(src_hist - ref_hist))
    print(f'cur_loss={cur_loss:.4f}, pre_loss={pre_loss:.4f}, loss下降了{((pre_loss - cur_loss) / pre_loss * 100):.2f}%')

    assert pre_loss - cur_loss > 0.0, 'Error!'
    print('Pass!')


# to do
def hist_spec(src_img, ref_img):
    """
    type src_img: ndarray
    type ref_img: ndarray
    rtype: ndarray
    """
    h1 = []
    p1 = []
    s1 = []
    S1 = []
    for i in range(256):
        h1.append(0)
    src_row, src_col = src_img.shape
    for i in range(src_row):
        for j in range(src_col):
            h1[src_img[i, j]] += 1
    for i in range(256):
        p1.append(h1[i] / src_img.size)
        s1.append(p1[0])
    for i in range(0, 255):
        s1.append(s1[i] + p1[i + 1])
    for i in range(256):
        S1.append(int(255 * s1[i - 1]))

    h2 = []
    p2 = []
    s2 = []
    S2 = []

    for i in range(256):
        h2.append(0)
    ref_row, ref_col = ref_img.shape
    for i in range(ref_row):
        for j in range(ref_col):
            h2[ref_img[i, j]] += 1
    for i in range(256):
        p2.append(h2[i] / ref_img.size)
        s2.append(p1[0])
    for i in range(0, 255):
        s2.append(s2[i] + p2[i + 1])
    for i in range(256):
        S2.append(round(255 * s2[i - 1]))

    r = []
    for i in range(256):
        m = S1[i]
        flag = True
        for j in range(256):
            if S2[j] == m:
                r.append(j)
                flag = False
                break
        if flag == True:
            nmin = 255
            for j in range(256):
                n = abs(S2[j] - m)
                if n < nmin:
                    nmin = n
                    jmin = j
            r.append(jmin)
    result = np.zeros_like(src_img)
    for i in range(src_row):
        for j in range(src_col):
            result[i, j] = r[src_img[i, j]]
    return result


if __name__ == '__main__':
    src_img = cv2.imread('E:/pythonProject/hw2/images/img_3.jpeg', flags=cv2.IMREAD_GRAYSCALE)  # 三张img的任意两两组合应该都Pass
    ref_img = cv2.imread('E:/pythonProject/hw2/images/img_2.jpeg', flags=cv2.IMREAD_GRAYSCALE)

    res = hist_spec(src_img, ref_img)

    plt.subplot(1, 3, 1)  # debug时可打开
    plt.imshow(src_img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(ref_img, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(res, cmap='gray')
    plt.show()

    judge(src_img, res, ref_img)
