import cv2 as cv
import math
import numpy as np
import scipy.stats as stats
import pandas as pd


def clear_list(data):
    data = pd.Series(data)
    Q1 = data.quantile(q=.25)
    Q3 = data.quantile(q=.75)
    IQR = data.apply(stats.iqr)

    # only keep rows in dataframe that have values within 1.5\*IQR of Q1 and Q3
    data_clean = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]
    res = data_clean.to_list()
    return res


def is_square(a, b, c, d):
    x1 = int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
    x2 = int(math.sqrt((b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2))
    x3 = int(math.sqrt((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2))
    x4 = int(math.sqrt((a[0] - d[0]) ** 2 + (a[1] - d[1]) ** 2))
    if abs(x1 - x2) < 2 and abs(x2 - x3) < 2 and abs(x3 - x4) < 2:
        return (x1 + x2 + x3 + x4) / 4
    return False


def getSize(filename):
    # image = cv.imread(filename)
    # b, g, r = cv.split(image)
    # filename = f'Images/WB{filename}'
    # cv.imwrite(filename, r)
    img = cv.imread(filename)
    hsv_min = np.array(
        (0, 20, 0),
        np.uint8)
    hsv_max = np.array(
        (255, 180, 255),
        np.uint8)

    height, width, _ = img.shape
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, hsv_min, hsv_max)
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mean = []
    for cnt in contours0:
        if len(cnt) < 150 and len(cnt) > 40:
            rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
            box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
            box = np.int0(box)  # округление координат
            a = [box[0][0], box[0][1]]
            b = [box[1][0], box[1][1]]
            c = [box[2][0], box[2][1]]
            d = [box[3][0], box[3][1]]
            res = is_square(a, b, c, d)
            if res:
                mean.append(res * 1.2)
                cv.drawContours(img, [box], 0, (255, 0, 0), 2)
    cv.imshow('contours', img)
    mean = clear_list(mean)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return sum(mean) / len(mean)


def getEllipse(filename):
    lst = filename.split('/')
    path = '/'.join(lst[0:-1])
    name = lst[-1]
    image = cv.imread(filename)
    b, g, r = cv.split(image)
    filename = f'{path}/WB{name}'
    cv.imwrite(filename, r)
    img = cv.imread(filename)
    hsv_min = np.array(
        (0, 0, 210),
        np.uint8)
    hsv_max = np.array(
        (255, 255, 255),
        np.uint8)

    height, width, _ = img.shape
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, hsv_min, hsv_max)
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt in contours0:
        if len(cnt) > 250:
            ellipse = cv.fitEllipse(cnt)
            coordx, coordy = ellipse[0]
            if coordx < width and coordx > 0 and coordy < height and coordy > 0:
                try:
                    res.append(ellipse)
                    cv.ellipse(img, ellipse, (0, 0, 255), 2)
                except:
                    continue
    cv.imshow('contours', img)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return res


if __name__ == "__main__":
    f_name = f'Images/3.jpg'
    one_cell = 1
    ellipse = getEllipse(f_name)
    size = getSize(f_name)
    one_px = one_cell / size
    w, h = ellipse[0][1]
    print(f'Круг с диаметром приблизительно {int((w * one_px + h * one_px) / 2)} мм')
