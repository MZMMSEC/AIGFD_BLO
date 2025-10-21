from skimage import io
from skimage import transform as sktransform
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import random
from PIL import Image
from imgaug import augmenters as iaa
import cv2, copy
import tqdm
from utils import *


def global_affine_transformation():
    # load image and the lmks
    image = io.imread(r"E:\CityuPC\_NextAIGC_Detection\Experiments\FDF\data\fdf\128\277184.png")
    face_img = sktransform.resize(image, (224, 224), preserve_range=True).astype(np.uint8)
    lmks = np.load(r"E:\CityuPC\_NextAIGC_Detection\Experiments\FAN\277184.npy")

    #
    distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.05, 0.06))])

    #
    # mask_un = parse(face_img, lmks[0], reg=3)
    mask = parse(face_img, lmks[0], reg=3, blurred=True)  # 1-3
    # mask_iaa = distortion.augment_image(mask)

    #
    img_iaa = distortion.augment_image(face_img)

    # blend
    out_img, _ = blend_func(face_img, img_iaa, mask)

    plt.imshow(img_iaa)
    plt.show()

    return out_img, img_iaa


def nose_flip():
    # load image and the lmks
    image = io.imread(r"E:\CityuPC\_NextAIGC_Detection\Experiments\FDF\data\fdf\128\277184.png")
    face_img = sktransform.resize(image, (224, 224), preserve_range=True).astype(np.uint8)
    lmks = np.load(r"E:\CityuPC\_NextAIGC_Detection\Experiments\FAN\277184.npy")
    mask_un = parse(face_img, lmks[0], reg=4)

    mask_un_color = np.zeros_like(face_img)
    mask_un_color[:, :, 0] = mask_un * 255
    mask_un_color[:, :, 1] = mask_un * 255
    mask_un_color[:, :, 2] = mask_un * 255

    top_ = round(15 * 224 / 128); size_bb_y = round(98 * 225 / 128); left_ = round(14 * 224 / 128); size_bb_x = round(98 * 224 / 128)
    plt.imshow(mask_un_color[top_: top_ + size_bb_y, left_: left_ + size_bb_x, :])
    plt.axis('off')
    plt.show()

    w, h, left, top, right, bottom = get_boundary(mask_un.astype(np.uint8))
    # region = face_img[top - 5:bottom + 5, left - 5:right + 5]
    region = face_img[top-3 :bottom+3 , left-3:right+3]

    # 生成一个矩形mask，based on region boundary
    rectangular_mask = np.zeros_like(face_img[:, :, 0])
    rectangular_mask[top:bottom, left:right] = 1

    rectangular_mask_color = np.zeros_like(face_img)
    rectangular_mask_color[:, :, 0] = rectangular_mask * 255
    rectangular_mask_color[:, :, 1] = rectangular_mask * 255
    rectangular_mask_color[:, :, 2] = rectangular_mask * 255

    plt.imshow(rectangular_mask_color[top_: top_ + size_bb_y, left_: left_ + size_bb_x, :])
    plt.axis('off')
    plt.show()
    ## guassian blur
    rectangular_mask = get_blend_mask(rectangular_mask, k1=15, k2=51, sig=5)
    rectangular_mask_color = np.zeros_like(face_img)
    rectangular_mask_color[:, :, 0] = rectangular_mask[:, :, 0] * 255
    rectangular_mask_color[:, :, 1] = rectangular_mask[:, :, 0] * 255
    rectangular_mask_color[:, :, 2] = rectangular_mask[:, :, 0] * 255

    plt.imshow(rectangular_mask_color[top_: top_ + size_bb_y, left_: left_ + size_bb_x, :])
    plt.axis('off')
    plt.show()

    # 利用 cv2.flip
    rotated_region = cv2.flip(region, -1)
    # rotated_region = cv2.flip(region, 1)
    face_img_rotate = copy.deepcopy(face_img)
    # face_img_rotate[top - 5:bottom + 5, left - 5:right + 5] = rotated_region
    face_img_rotate[top-3 :bottom+3 , left-3:right+3] = rotated_region

    plt.imshow(face_img_rotate[top_: top_ + size_bb_y, left_: left_ + size_bb_x, :])
    plt.axis('off')
    plt.show()

    out_img, _ = blend_func(face_img, face_img_rotate, rectangular_mask)

    plt.imshow(out_img.astype(int)[top_: top_ + size_bb_y, left_: left_ + size_bb_x, :])
    plt.axis('off')
    plt.show()


def mouth_flip(flip_code=1): # or -1
    # load image and the lmks
    image = io.imread(r"E:\CityuPC\_NextAIGC_Detection\Experiments\FDF\data\fdf\128\277184.png")
    face_img = sktransform.resize(image, (224, 224), preserve_range=True).astype(np.uint8)
    lmks = np.load(r"E:\CityuPC\_NextAIGC_Detection\Experiments\FAN\277184.npy")

    mask_un = parse(face_img, lmks[0], reg=3)

    mask_un_color = np.zeros_like(face_img)
    mask_un_color[:, :, 0] = mask_un * 255
    mask_un_color[:, :, 1] = mask_un * 255
    mask_un_color[:, :, 2] = mask_un * 255

    top_ = round(15*224/128); size_bb_y = round(98*225/128); left_ = round(14*224/128); size_bb_x = round(98*224/128)
    plt.imshow(mask_un_color[top_: top_+size_bb_y, left_: left_+size_bb_x, :])
    plt.axis('off')
    plt.show()

    w, h, left, top, right, bottom = get_boundary(mask_un.astype(np.uint8))
    region = face_img[top-5:bottom+5, left-5:right+5]

    # generate rectangular mask，based on region boundary
    rectangular_mask = np.zeros_like(face_img[:, :, 0])
    rectangular_mask[top:bottom, left:right] = 1

    rectangular_mask_color = np.zeros_like(face_img)
    rectangular_mask_color[:, :, 0] = rectangular_mask * 255
    rectangular_mask_color[:, :, 1] = rectangular_mask * 255
    rectangular_mask_color[:, :, 2] = rectangular_mask * 255

    plt.imshow(rectangular_mask_color[top_: top_+size_bb_y, left_: left_+size_bb_x, :])
    plt.axis('off')
    plt.show()
    ## guassian blur
    rectangular_mask = get_blend_mask(rectangular_mask, k1=91, k2=51, sig=15)
    rectangular_mask_color = np.zeros_like(face_img)
    rectangular_mask_color[:, :, 0] = rectangular_mask[:,:,0] * 255
    rectangular_mask_color[:, :, 1] = rectangular_mask[:,:,0] * 255
    rectangular_mask_color[:, :, 2] = rectangular_mask[:,:,0] * 255

    plt.imshow(rectangular_mask_color[top_: top_+size_bb_y, left_: left_+size_bb_x, :])
    plt.axis('off')
    plt.show()

    # 利用 cv2.flip
    rotated_region = cv2.flip(region, flipCode=flip_code)
    face_img_rotate = copy.deepcopy(face_img)
    face_img_rotate[top-5:bottom+5, left-5:right+5] = rotated_region

    plt.imshow(face_img_rotate[top_: top_+size_bb_y, left_: left_+size_bb_x, :])
    plt.axis('off')
    plt.show()


    out_img, _ = blend_func(face_img, face_img_rotate, rectangular_mask)

    plt.imshow(out_img.astype(int)[top_: top_+size_bb_y, left_: left_+size_bb_x, :])
    plt.axis('off')
    plt.show()


def eye_flip():
    # load image and the lmks
    image = io.imread(r"E:\CityuPC\_NextAIGC_Detection\Experiments\FDF\data\fdf\128\277184.png")
    face_img = sktransform.resize(image, (224, 224), preserve_range=True).astype(np.uint8)
    lmks = np.load(r"E:\CityuPC\_NextAIGC_Detection\Experiments\FAN\277184.npy")

    mask_un = parse(face_img, lmks[0], reg=0)
    w, h, left, top, right, bottom = get_boundary(mask_un.astype(np.uint8))

    mask_un_color = np.zeros_like(face_img)
    mask_un_color[:, :, 0] = mask_un * 255
    mask_un_color[:, :, 1] = mask_un * 255
    mask_un_color[:, :, 2] = mask_un * 255
    mask_un_color[top - 5:bottom + 5, left - 5:right + 5, :] = 255
    plt.imshow(mask_un_color)
    plt.show()

    region = face_img[top - 5:bottom + 5, left - 5:right + 5]
    plt.imshow(region)
    plt.show()

    #
    rectangular_mask = np.zeros_like(face_img[:, :, 0])
    rectangular_mask[top:bottom, left:right] = 1
    ## guassian blur
    rectangular_mask = get_blend_mask(rectangular_mask, k1=91, k2=91, sig=15)
    plt.imshow(rectangular_mask)
    plt.show()


    rotated_region = cv2.flip(region, 1)
    face_img_rotate = copy.deepcopy(face_img)
    face_img_rotate[top - 5:bottom + 5, left - 5:right + 5] = rotated_region

    out_img, _ = blend_func(face_img, face_img_rotate, rectangular_mask)

    plt.imshow(face_img)
    plt.show()
    plt.imshow(face_img_rotate)
    plt.show()
    plt.imshow(out_img.astype(int))
    plt.show()


if __name__ == '__main__':
    global_affine_transformation()