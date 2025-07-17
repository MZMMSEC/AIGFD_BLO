import os
import pdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch, cv2
import random
import json
from tqdm import tqdm


def _transform(res):
    return Compose([
        ToTensor(),
        Resize((res,res)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def bbox_crop_pil2cvt2pil(neg_imgpath, meta_data_dict, scale_factor=1.3):
    # pil loading
    input = Image.open(neg_imgpath).convert('RGB')
    # transform to cv2
    input = cv2.cvtColor(np.asarray(input),cv2.COLOR_RGB2BGR)
    # get the metadata based on path name
    if '_' in neg_imgpath: # first negative samples
        name_id = neg_imgpath.split('/')[-1].split('_')[0]
    else: # otherwise, positive samples
        name_id = neg_imgpath.split('/')[-1].split('.')[0]
    meta_data = meta_data_dict[name_id]

    left, top = round(meta_data['bounding_box'][0] * input.shape[0]), round(meta_data['bounding_box'][1] * input.shape[0])
    right, bottom = round(meta_data['bounding_box'][2] * input.shape[0]), round(meta_data['bounding_box'][3] * input.shape[0])
    size_bb = int(max(right - left, bottom - top) * scale_factor)
    center_x, center_y = (right + left) // 2, (bottom + top) // 2
    # 按scale扩大的bounding box的left top
    x1_ = max(int(center_x - size_bb // 2), 0)
    y1_ = max(int(center_y - size_bb // 2), 0)
    # 扩大的bounding box的size,但得考虑bounding box不能超出图像尺寸
    height, width = input.shape[0], input.shape[1]
    size_bb_x = min(width - x1_, size_bb)
    size_bb_y = min(height - y1_, size_bb)
    left, top = x1_, y1_

    cropped_face = input[top: top + size_bb_y, left: left + size_bb_x, :]
    cropped_pil_image = Image.fromarray(np.uint8(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)))
    return cropped_pil_image

def random_select_index(lst, train_mode=True):
    # index = random.randint(0, len(lst) - 1)
    if train_mode:
        index = random.choice([0,0,0,0,1,2,3,4])
    else:
        index = 0
    selected_element = lst[index]
    return index, selected_element

class selectiveEXIF_data(Dataset):
    def __init__(self, meta_json="/data0/mian3-2/Experiments/exif-as-language-L2R-v2-debugExp/EXIF_study/fdf_ccby2_exif_ordinal_categorical.json",
                 face_root_path='/data0/mian3-2/Experiments/FDF/data/fdf/images/128',
                 neg_queue_root_path = '/data0/mian3-2/Experiments/FAN/face_dataaug_neg',
                 metadata=None, face_metainfo_root_path='/data0/mian3-2/Experiments/FDF/data/fdf/fdf_metainfo.json',
                 resolution=224, face_scale=1.3, train_mode=True, cls_face_mode=False
                 ):
        super(selectiveEXIF_data, self).__init__()
        self.transforms = _transform(resolution)
        self.neg_queue_root_path = neg_queue_root_path
        self.train_mode = train_mode
        self.res = resolution
        self.face_scale = face_scale
        self.cls_face_mode = cls_face_mode
        self.imgs = []
        self.get_all_items(meta_json, face_root_path, neg_queue_root_path)

        self.imgs_use = self.split_train_val(train_mode)
        print(f'total length of the dataset is {len(self.imgs_use)} faces...')

        if face_metainfo_root_path is None:
            self.metadata = metadata
        else:
            with open(face_metainfo_root_path) as file:
                self.metadata = json.load(file)

    def split_train_val(self, train=True):
        random.shuffle(self.imgs)

        if train:
            imgs_use = self.imgs[:-5000]
        else:
            imgs_use = self.imgs[-5000:]

        return imgs_use

    def __len__(self):
        return len(self.imgs_use)

    def get_all_items(self, meta_json, face_root_path, neg_queue_root_path):
        with open(meta_json) as file:
            metadata = json.load(file)
        print(f"total faces collected: {len(metadata)}....")

        for item in tqdm(metadata):
            face_exif_items = {}
            face_path_list = []
            face_path = metadata[item]['path']

            face_path_list.append(face_path)
            if self.train_mode:
                neg_queue_path_base = face_path.replace(face_root_path, neg_queue_root_path).split('.')[0]
                if not os.path.isfile(neg_queue_path_base + '_0.png'):
                    continue
                neg_queue_paths = []
                for num in range(4):
                    neg_queue_paths.append(
                        neg_queue_path_base + '_' + str(num) + '.png'
                    )
                face_path_list.extend(neg_queue_paths)

            face_exif_items['imgpath'] = face_path_list
            face_exif_items['EXIF_categorical'] = metadata[item]['EXIF_categorical']
            face_exif_items['EXIF_ordinal'] = metadata[item]['EXIF_ordinal']
            self.imgs.append(face_exif_items)

    def __getitem__(self, idx):
        face_exif_items = self.imgs_use[idx]
        label = {}
        ## get EXIF ordinal
        # pdb.set_trace()
        label['iso'] = face_exif_items['EXIF_ordinal']['ISO']
        label['av'] = face_exif_items['EXIF_ordinal']['Aperture']
        label['fl'] = face_exif_items['EXIF_ordinal']['Focal Length']
        label['et'] = face_exif_items['EXIF_ordinal']['Exposure Time']
        ## get EXIF categorical
        label['makes'] = face_exif_items['EXIF_categorical']['Make']['cls_label']
        label['mm'] = face_exif_items['EXIF_categorical']['Metering Mode']['cls_label']
        label['em'] = face_exif_items['EXIF_categorical']['Exposure Mode']['cls_label']
        label['wb'] = face_exif_items['EXIF_categorical']['White Balance Mode']['cls_label']
        label['ep'] = face_exif_items['EXIF_categorical']['Exposure Program']['cls_label']

        # get img path
        imgpath_ls = face_exif_items['imgpath']
        img_idx, imgpath = random_select_index(imgpath_ls, train_mode=self.train_mode)
        # get img
        img = bbox_crop_pil2cvt2pil(imgpath, self.metadata, scale_factor=self.face_scale)
        img = self.transforms(img)
        if self.cls_face_mode:
            # get cls label for face
            if img_idx == 0:
                label['face2text'] = torch.tensor([1, 0, 0, 0])  # regular
            elif img_idx == 1:
                label['face2text'] = torch.tensor([0, 1, 1, 0])  # irregular mouth
            elif img_idx == 2:
                label['face2text'] = torch.tensor([0, 1, 1, 0])  # irregular mouth
            elif img_idx == 3:
                label['face2text'] = torch.tensor([0, 1, 0, 1])  # irregular eyes
            else:
                label['face2text'] = torch.tensor([0, 1, 1, 1])  # irregular whole face

        return img, label

if __name__ == '__main__':
    selectiveEXIF_data()