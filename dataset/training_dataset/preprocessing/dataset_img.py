from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop
import glob, pickle, os, json, imageio, random
from tqdm import tqdm
from skimage import transform as sktransform
import torch
import numpy as np

from utils import *

def _transform(res):
    return Compose([
        # _convert_image_to_rgb,
        ToTensor(),
        # RandomCrop((res,res)),
        Resize((res,res)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class yfcc_face(Dataset):
    def __init__(self, face_root_path='/data0/mian3-2/Experiments/FDF/data/fdf/images/128',
                 exif_root_path="/data0/mian3-2/Experiments/yfcc-process/save/fdf_ccby2_exif_update_filtered_v2/",
                 photoid_vs_imgname='/data0/mian3-2/Experiments/FDF/data/id_vs_fdfName.pkl',
                 lmks_root_path='/data0/mian3-2/Experiments/yfcc-lmks-224',
                 resolution=224, train_mode=True):
        super(yfcc_face, self).__init__()
        self.face_root_path = face_root_path
        self.exif_root_path = exif_root_path
        self.photoid_vs_imgname = photoid_vs_imgname
        self.lmks_root_path = lmks_root_path
        self.transforms = _transform(resolution)
        self.res = resolution

        self.imgs = []

        self.gather_possible_items(self.face_root_path, self.exif_root_path,
                                   self.photoid_vs_imgname, self.lmks_root_path)

        self.imgs_use = self.split_train_val(train_mode)

        print(f'total length of the dataset is {len(self.imgs_use)} faces...')

    def __len__(self):
        return len(self.imgs_use)

    def __getitem__(self, idx):
        face_exif_items = self.imgs_use[idx]

        # get img
        imgpath = face_exif_items['imgpath']
        ## for negative queue
        lmks_path = face_exif_items['lmk_path']

        ## 根据lmks去构建K个负样本
        neg_samples = self.lmks_operator(lmks_path, imgpath)
        neg_queue = []
        for num, neg_im in enumerate(neg_samples):
            neg_im = neg_im.resize((128, 128),Image.BILINEAR)
            img_name = imgpath.split('/')[-1].split('.')[0]
            savepath = os.path.join("/data0/mian3-2/Experiments/FAN/face_dataaug_neg/", img_name+'_'+str(num)+'.png')
            neg_im.save(savepath)

        return imgpath

    def lmks_operator(self, lmks_path, face_path):
        lmks = np.load(lmks_path, allow_pickle=True)
        # first resize the face to 224x224
        face_img = imageio.imread(face_path)
        face_img = sktransform.resize(face_img, (224, 224), preserve_range=True).astype(np.uint8)

        if lmks.shape == ():
            mouth_hflip = face_distortion(face_img, lmks)
            mouth_vflip = face_distortion(face_img, lmks)
            eye_hflip = face_distortion(face_img, lmks)
            face_distort = face_distortion(face_img, lmks)
        else:
            mouth_hflip = mouth_flip(face_img, lmks, flip_type=1)  # mouth horizontal flip
            if mouth_hflip is None:
                mouth_hflip = face_distortion(face_img, lmks)
                mouth_vflip = face_distortion(face_img, lmks)
            else:
                mouth_vflip = mouth_flip(face_img, lmks, flip_type=-1)  # mouth vertical flip
            eye_hflip = eye_flip(face_img, lmks)  # eye horizontal flip
            if eye_hflip is None:
                eye_hflip = face_distortion(face_img, lmks)
            face_distort = face_distortion(face_img, lmks)  # global affine transformation

        return [mouth_vflip, mouth_hflip, eye_hflip, face_distort]

    def gather_possible_items(self, face_root_path, exif_root_path, photoid_vs_imgname, lmks_root_path):
        face_paths = glob.glob(f"{face_root_path}/*.png")
        with open(photoid_vs_imgname, 'rb') as file:
            photoid_vs_imgname_dict = pickle.load(file)

        for face in tqdm(face_paths):
            lmks_path = face.replace(face_root_path, lmks_root_path).replace('.png', '.npy')

            if not os.path.isfile(lmks_path):
                continue

            # get the EXIF tag information
            imgname = face.split('/')[-1].split('.')[0]
            photo_id = photoid_vs_imgname_dict[imgname]
            exif_path = os.path.join(exif_root_path, photo_id + '.json')
            if not os.path.isfile(exif_path):
                continue

            exif_info, exif_info_str, exif_num = self.get_exif_str(exif_path)
            if exif_info is None:
                continue

            face_exif_items = {}
            face_exif_items['imgpath'] = face
            face_exif_items['exif_num'] = exif_num
            face_exif_items['exif_info_str'] = exif_info_str
            face_exif_items['lmk_path'] = lmks_path
            self.imgs.append(face_exif_items)

    def split_train_val(self, train=True):
        random.shuffle(self.imgs)

        if train:
            imgs_use = self.imgs[:-5000]
        else:
            imgs_use = self.imgs[-5000:]

        return imgs_use

    def get_exif_str(self, exif_path):
        with open(exif_path, 'rb') as file:
            exif = json.load(file)

            exif_info = {}
            exif_info['ISO Speed Ratings'] = exif['EXIF']['ISO Speed Ratings']
            exif_info['Aperture Value'] = exif['EXIF']['Aperture Value']
            exif_info['Exposure Time'] = exif['EXIF']['Exposure Time']
            exif_info['Focal Length'] = exif['EXIF']['Focal Length']

            if exif_info['Aperture Value'] == 'F':
                return None, None, None

            exif_num = self.get_exif_statistics(exif_info)

            exif_str = ", ".join([f"{key}: {value}" for key, value in exif_info.items()])

            return exif_info, exif_str, exif_num

    def get_exif_statistics(self, exif_info):
        try:
            ISO = float(exif_info['ISO Speed Ratings'])
            AV = float(exif_info['Aperture Value'].split('F')[-1])
            FL = float(exif_info['Focal Length'].split(' mm')[0])
            ET = float(exif_info['Exposure Time'].split('sec')[0].split('1/')[-1])
        except:
            ISO = float(exif_info['ISO Speed Ratings'])
            AV = float(eval(exif_info['Aperture Value'].split('F')[-1]))
            FL = float(exif_info['Focal Length'].split(' mm')[0])
            ET = float(eval(exif_info['Exposure Time'].split('sec')[0]))

        exif_num = {}
        exif_num['ISO Speed Ratings'] = ISO
        exif_num['Aperture Value'] = AV
        exif_num['Focal Length'] = FL
        exif_num['Exposure Time'] = ET
        return exif_num


def setup_seed(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    setup_seed(42)
    dataset = yfcc_face(train_mode=True)

    train_dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    for idx, batch in enumerate(tqdm(train_dataloader)):
        continue