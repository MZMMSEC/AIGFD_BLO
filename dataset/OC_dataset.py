import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop



def _transform(res):
    return Compose([
        # _convert_image_to_rgb,
        ToTensor(),
        # RandomCrop((res,res)),
        Resize((res,res)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class GatherDataset(Dataset):
    def __init__(self, real_item_tag='celeba',
                 only_real=False, only_fake=False,
                 fake_items_tag='all',
                 training=True,
                 random_subset=False,
                 random_subset_num_fake=200,
                 random_subset_num_real=800,
                 resize_size=224
                 ):
        self.image_paths = []
        self.transforms = _transform(resize_size)


        if fake_items_tag == 'all':
            fake_items = [
                'stylegan2', 'vqgan',
                'ldm', 'ddim', 'sdv21',
                'freeDom', 'hps', 'midj', 'sdxl']
        else:
            fake_items = [fake_items_tag]

        root_dir_fake_dict = {
            'progan':'../data/progan/',
            'stargan': '../data/stargan/',
            'pigan':'../data/pigan/',
            #
            'stylegan2': '../data/stylegan2/',#
            'vqgan': "../data/vqgan/",
            'ldm': "../data/ldm/",
            'ddim': "../data/ddim/",
            'sdv21': "../data/sdv21/",
            'freeDom': "../data/freeDoM/",
            'hps': "../data/hps/",
            'midj': "../data/midjourney/",
            'sdxl': "../data/sdxl/"
        }
        self.image_paths_fake = []
        for i, fake_item in enumerate(fake_items):
            self.image_paths_current = []
            root_dir_fake = root_dir_fake_dict[fake_item]
            self._collect_image_paths_fake(root_dir_fake, i)
            if random_subset:
                if random_subset_num_fake < len(self.image_paths_current):
                    self.image_paths_current = random.sample(self.image_paths_current, random_subset_num_fake)

            self.image_paths_fake.extend(self.image_paths_current)

        length_fake = len(self.image_paths_fake)

        # for photographic
        real_items = [real_item_tag]
        root_dir_real_dict = {
            'celeba': "../data/celeba-face/",
        }
        self.image_paths_real = []
        i = 8
        for real_item in real_items:
            self.image_paths_current = []
            if real_item in ['celeba']:
                self.collect_celeba(i,training)
            else:
                root_dir_real = root_dir_real_dict[real_item]
                self._collect_image_paths_real(root_dir_real, i)

            print(len(self.image_paths_current))
            i = i + 1
            if random_subset:
                image_paths_tmp = self.image_paths_current

                if training:
                    self.image_paths_current = image_paths_tmp[:random_subset_num_real]
                else:
                    self.image_paths_current = random.sample(image_paths_tmp, length_fake)

            self.image_paths_real.extend(self.image_paths_current)

        if not only_real and not only_fake:
            self.image_paths.extend(self.image_paths_real)
            self.image_paths.extend(self.image_paths_fake)
        elif only_fake:
            self.image_paths.extend(self.image_paths_fake)
        else:
            self.image_paths.extend(self.image_paths_real)

    def collect_celeba(self, label_specific, train=True):
        if train:
            print('training!')
            with open(
                    "../data/celeba-face/train.txt",
                    'r') as file:
                for line in file:
                    words = line.strip()
                    file_path = words
                    self.image_paths_current.append((file_path, 0, label_specific))
        else:
            with open(
                    "../data/celeba-face/test.txt",
                    'r') as file:
                for line in file:
                    words = line.strip()
                    file_path = words
                    self.image_paths_current.append((file_path, 0, label_specific))


    def _collect_image_paths_fake(self, dir_path, label_specific):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isdir(file_path):
                self._collect_image_paths_fake(file_path, label_specific)
            elif filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                self.image_paths_current.append((file_path, 1, label_specific))

    def _collect_image_paths_real(self, dir_path, label_specific):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isdir(file_path):
                self._collect_image_paths_real(file_path, label_specific)
            elif filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                self.image_paths_current.append((file_path, 0, label_specific))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label, label_specific = self.image_paths[idx]
        # img_name = img_path.split('/')[-1].split('.')[0]

        # Get image
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)

        samples = {}
        samples['img'] = img
        samples['label'] = label
        samples['label_specific'] = label_specific
        samples['path'] = img_path

        return samples

