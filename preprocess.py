import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch

from PIL import Image
import urllib.request
import os


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import json
import os
import pickle
from PIL import Image
import cv2

from simple_tokenizer import SimpleTokenizer


# reference
# https://visualgenome.org/api/v0/api_home.html
class Custom(Dataset):
    def __init__(self, args, transform=None):
        filename = os.path.join(args.base_dir, 'data.pkl')
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)

        self.image_path = os.path.join(args.base_dir, 'data')

        self.transform = transform
        self.tokenizer = SimpleTokenizer()
        self.max_len = args.max_len

    def __getitem__(self, idx):
        img_name, init_text = self.data[idx]['image_path'], self.data[idx]['phrase']
        img_filename = os.path.join(self.image_path, img_name)

        img = cv2.imread(img_filename)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        text = [sot_token] + self.tokenizer.encode(init_text) + [eot_token]

        text = text[:self.max_len]
        padding = [0 for _ in range(self.max_len - len(text))]

        text += padding
        text = torch.tensor(text)

        return img, text

    def __len__(self):
        return len(self.data.keys())


def prepare_data(args):
    img_dir = os.path.join(args.base_dir, 'VG_100K_2')
    annotation_filename = os.path.join(args.base_dir, 'region_descriptions.json')

    with open(annotation_filename, 'r') as f:
        data = json.load(f)

    save_dir = os.path.join(args.base_dir, 'data')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    data_dict, count = {}, 0
    for idx, _ in enumerate(data):
        image_path = os.path.join(img_dir, '{}.jpg'.format(_['regions'][0]['image_id']))
        if os.path.isfile(image_path):
            img = cv2.imread(image_path)
            for i in range(len(_['regions'])):
                width, height, image_id, phrase, y, x = _['regions'][i]['width'], _['regions'][i]['height'],\
                                                        _['regions'][i]['image_id'], _['regions'][i]['phrase'],\
                                                        _['regions'][i]['y'], _['regions'][i]['x']

                if x < 0:
                    x = 0
                if y < 0:
                    y = 0

                try:
                    crop_img = img[y:y + height, x:x + width, :]
                    crop_img = cv2.resize(crop_img, (32, 32))
                    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format(count)), crop_img)

                    data_dict[count] = {
                        'image_path': '{}.jpg'.format(count),
                        'phrase': phrase
                    }
                    count += 1
                except:
                    pass

    with open(os.path.join(args.base_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


def load_data(args, eval_data='CIFAR10'):
    if args.dataset == 'CIFAR10':
        test_transform = transforms.Compose([
            transforms.Resize(args.input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_transform = transforms.Compose([
            transforms.Resize(args.input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        train_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'CIFAR100':
        test_transform = transforms.Compose([
            transforms.Resize(args.input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_transform = transforms.Compose([
            transforms.Resize(args.input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        train_data = torchvision.datasets.CIFAR100(root='./data', train=False, transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        # custom data (wip)
        if not os.path.isfile(os.path.join(args.base_dir, 'data.pkl')):
            print('Prepare Data...')
            prepare_data(args)
        else:
            print('Data Ready...')

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = Custom(args, test_transform)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        train_data = Custom(args, train_transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_data, train_loader, test_data, test_loader
