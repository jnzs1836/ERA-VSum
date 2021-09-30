# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path
from torchvision import transforms

import h5py
import numpy as np
import os


class FrameData(Dataset):
    def __init__(self, root, transform, frame_range=[-1, -1]):
        self.frame_dir = root
        self.transform = transform
        frame_list = os.listdir(root)
        frame_items = list(map(lambda x: (FrameData._extract_file_id(x), x), frame_list))
        frame_items.sort(key=lambda x: x[0])
        self.frame_files = list(map(lambda x: x[1], frame_items))
        if frame_range[0] == -1:
            pass
        else:
            self.frame_files = self.frame_files[frame_range[0]:frame_range[1]]
        # print()

    @staticmethod
    def _extract_file_id(filename):
        idx = filename[3:-4]
        idx = int(idx)
        return idx

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, index):
        image_file = self.frame_files[index]
        image_path = os.path.join(self.frame_dir, image_file)
        img = default_loader(image_path)
        image_tensor = self.transform(img)
        return image_tensor, image_file


class VideoData(Dataset):
    def __init__(self, root, preprocessed=True, transform=None, with_name=False):
        self.root = root
        self.transform = transform
        frame_list = os.listdir(root)
        frame_items = list(map(lambda x: (VideoData._extract_file_id, x), frame_list))
    @staticmethod
    def _extract_file_id(filename):
        idx = filename[3:-4]
        idx = int(idx)
        return idx

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        images = []
        count = 0
        for img_path in Path(self.video_list[index]).glob('*.jpg'):
            img = default_loader(img_path)
            img_tensor = self.transform(img)
            images.append(img_tensor)
            count += 1
            if count == 256:
                break
        print(images[0].size())
        return torch.stack(images), img_path.parent.name[4:]


def get_frame_loader(root, batch_size=32, frame_range=[-1, -1]):
    transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return DataLoader(FrameData(root, transform=transform, frame_range=frame_range), batch_size=batch_size)


if __name__ == '__main__':
    pass
