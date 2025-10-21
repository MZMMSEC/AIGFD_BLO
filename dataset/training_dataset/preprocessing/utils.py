import pdb
import imageio, cv2, math, copy
from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np
import random
from PIL import Image, ImageDraw
from imgaug import augmenters as iaa

def nose_flip_func(face_img, lmks, flip_type=-1):
    try:
        mask_un = parse(face_img, lmks[0], reg=4)
    except:
        pdb.set_trace()
    w, h, left, top, right, bottom = get_boundary(mask_un.astype(np.uint8))
    if w is None:
        return None

    top_region = max(top - 3, 0)
    bottom_region = min(bottom + 3, mask_un.shape[0])
    left_region = max(left - 3, 0)
    right_region = min(right + 3, mask_un.shape[0])
    region = face_img[top_region:bottom_region, left_region:right_region]

    rectangular_mask = np.zeros_like(face_img[:, :, 0])
    rectangular_mask[top:bottom, left:right] = 1
    ## guassian blur
    rectangular_mask = get_blend_mask(rectangular_mask, k1=15, k2=51, sig=5)

    rotated_region = cv2.flip(region, flip_type)  # vertical: -1, horizontal: 1
    face_img_rotate = copy.deepcopy(face_img)
    try:
        face_img_rotate[top_region:bottom_region, left_region:right_region] = rotated_region
    except:
        pdb.set_trace()

    out_img, _ = blend_func(face_img, face_img_rotate, rectangular_mask)
    return Image.fromarray(np.uint8(out_img))


def mouth_flip(face_img, lmks, flip_type=-1):
    try:
        mask_un = parse(face_img, lmks[0], reg=3)
    except:
        pdb.set_trace()
    w, h, left, top, right, bottom = get_boundary(mask_un.astype(np.uint8))
    if w is None:
        return None
    top_region = max(top - 5, 0)
    bottom_region = min(bottom + 5, mask_un.shape[0])
    left_region = max(left - 5, 0)
    right_region = min(right + 5, mask_un.shape[0])
    region = face_img[top_region:bottom_region, left_region:right_region]

    rectangular_mask = np.zeros_like(face_img[:, :, 0])
    rectangular_mask[top:bottom, left:right] = 1
    ## guassian blur
    rectangular_mask = get_blend_mask(rectangular_mask, k1=91, k2=51, sig=15)

    rotated_region = cv2.flip(region, flip_type) # vertical: -1, horizontal: 1
    face_img_rotate = copy.deepcopy(face_img)
    try:
        face_img_rotate[top_region:bottom_region, left_region:right_region] = rotated_region
    except:
        pdb.set_trace()

    out_img, _ = blend_func(face_img, face_img_rotate, rectangular_mask)
    return Image.fromarray(np.uint8(out_img))

def eye_flip(face_img, lmks):
    mask_un = parse(face_img, lmks[0], reg=1)
    w, h, left, top, right, bottom = get_boundary(mask_un.astype(np.uint8))
    if w is None:
        return None
    top_region = max(top - 5, 0)
    bottom_region = min(bottom + 5, mask_un.shape[0])
    left_region = max(left - 5, 0)
    right_region = min(right + 5, mask_un.shape[0])
    region = face_img[top_region:bottom_region, left_region:right_region]

    rectangular_mask = np.zeros_like(face_img[:, :, 0])
    rectangular_mask[top:bottom, left:right] = 1
    ## guassian blur
    rectangular_mask = get_blend_mask(rectangular_mask, k1=91, k2=91, sig=15)

    rotated_region = cv2.flip(region, 1)
    face_img_rotate = copy.deepcopy(face_img)
    face_img_rotate[top_region:bottom_region, left_region:right_region] = rotated_region

    out_img, _ = blend_func(face_img, face_img_rotate, rectangular_mask)
    return Image.fromarray(np.uint8(out_img))

def face_distortion(face_img, lmks):
    distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.04, 0.05))])

    img_iaa = distortion.augment_image(face_img)

    # reg_list = [1, 2, 2, 3, 3, 3]
    # reg = reg_list[np.random.randint(len(reg_list))]
    # mask = parse(face_img, lmks[0], reg, blurred=True)
    # out_img, _ = blend_func(face_img, img_iaa, mask)
    # return Image.fromarray(np.uint8(out_img)), Image.fromarray(np.uint8(img_iaa))
    return Image.fromarray(np.uint8(img_iaa))


def get_boundary(mask):
    # 查找掩码的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取第一个轮廓
    try:
        contour = contours[0]
    except:
        return None, None, None, None, None, None

    # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)

    # 计算边界坐标
    left = x
    top = y
    right = x + w
    bottom = y + h

    return w, h, left, top, right, bottom

def parse(img, real_lmk, reg, blurred=False):
    five_key = get_five_key(real_lmk)
    if reg == 0:
        mask = remove_eyes(img, five_key, 'l')
    elif reg == 1:
        mask = remove_eyes(img, five_key, 'r')
    elif reg == 2:
        # mask = remove_eyes(img, five_key, 'b')
        mask = remove_eyes(img, five_key, 'l') + remove_eyes(img, five_key, 'r')
    elif reg == 3:
        mask = remove_mouth(img, five_key)
    elif reg == 4: # nose
        mask = remove_nose(img, five_key)

    if blurred:
        mask = get_blend_mask(mask.astype(int).astype(float)) # blurred mask

    # img_aug, _ = blend_func(img, img, mask)

    return mask

def blend_func(realimg, fakeimg, deformed_fakemask):
    H, W, C = realimg.shape

    aligned_src = fakeimg

    # src_mask = deformed_fakemask > 0  # (H, W)

    # tgt_mask = np.asarray(src_mask)
    tgt_mask = deformed_fakemask
    # mask post-processing
    # tgt_mask = mask_postprocess(tgt_mask)  # erode or dilate the mask before Gaussian blur

    # color transfer
    # aligned_src = colorTransfer_func(realimg, aligned_src, tgt_mask * 255)

    out_blend, tgt_mask = dynamic_blend(aligned_src, realimg, tgt_mask.astype(float))
    return out_blend, tgt_mask

def dynamic_blend(source,target,mask_blured):
    blend_list=[0.75,1,1,1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]

    # blend_ratio = 1
    mask_blured*=blend_ratio
    if len(mask_blured.shape) == 2:
        mask_blured = np.expand_dims(mask_blured, 2).repeat(3, axis=2)
    # import matplotlib.pyplot as plt
    # plt.imshow((mask_blured * source).astype(int)); plt.show()
    # plt.imshow(((1 - mask_blured) * target).astype(int)); plt.show()
    img_blended=(mask_blured * source + (1 - mask_blured) * target)
    return img_blended,mask_blured

def get_blend_mask(mask, k1=5, k2=15, sig=0):
    H, W = mask.shape
    # size_h = np.random.randint(192, 257)
    # size_w = np.random.randint(192, 257)
    # mask = cv2.resize(mask, (size_w, size_h))
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)

    kernel_1 = (k1, k1)
    kernel_2 = (k2, k2)
    mask_blured = cv2.GaussianBlur(mask, kernel_1, sig//2)
    if mask_blured.max() == 0:
        mask_blured = cv2.GaussianBlur(mask, kernel_1, 1)
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured[mask_blured < 1] = 0

    eroded_mask = cv2.erode(mask_blured, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    if eroded_mask.max() == 0:
        eroded_mask = mask_blured
    mask_blured = cv2.GaussianBlur(eroded_mask, (51, 51), 10)
    if mask_blured.max() == 0:
        mask_blured = cv2.GaussianBlur(eroded_mask, (51, 51), 0)

    # mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, sig) #np.random.randint(5, 46))
    if mask_blured.max() == 0:
        pdb.set_trace()
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured = cv2.resize(mask_blured, (W, H))
    return mask_blured.reshape((mask_blured.shape + (1,)))

def get_five_key(landmarks_68):
    # get the five key points by using the landmarks
    leye_center = (landmarks_68[36] + landmarks_68[39])*0.5
    reye_center = (landmarks_68[42] + landmarks_68[45])*0.5
    nose = landmarks_68[33]
    lmouth = landmarks_68[48]
    rmouth = landmarks_68[54]
    leye_left = landmarks_68[36]
    leye_right = landmarks_68[39]
    reye_left = landmarks_68[42]
    reye_right = landmarks_68[45]
    out = [ tuple(x.astype('int32')) for x in [
        leye_center,reye_center,nose,lmouth,rmouth,leye_left,leye_right,reye_left,reye_right
    ]]
    return out

def remove_eyes(image, landmarks, opt):
    ##l: left eye; r: right eye, b: both eye
    if opt == 'l':
        (x1, y1), (x2, y2) = landmarks[5:7]
    elif opt == 'r':
        (x1, y1), (x2, y2) = landmarks[7:9]
    elif opt == 'b':
        (x1, y1), (x2, y2) = landmarks[:2]
    else:
        print('wrong region')
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    if opt != 'b':
        dilation *= 2 #4
    line = binary_dilation(line, iterations=dilation)
    return line

def remove_mouth(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[3:5]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    # pdb.set_trace()
    return line

def remove_nose(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    return line

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_boundary(mask):
    # 查找掩码的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取第一个轮廓
    contour = contours[0]

    # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)

    # 计算边界坐标
    left = x
    top = y
    right = x + w
    bottom = y + h
    return w, h, left, top, right, bottom
