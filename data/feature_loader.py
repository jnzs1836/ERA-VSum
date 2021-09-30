# -*- coding: utf-8 -*-
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import h5py
import numpy as np
import os
from .frame_loader import FrameData
from deployment.io import read_video_sample

class VideoFeatureDataset(Dataset):
    def __init__(self, h5path, splits=None, transform=None, with_images=False, image_dir=None, video_dir=None, mapping_file_path=None):
        self.h5path = h5path
        self.transform = transform
        # self.with_name = with_name
        self.hf = h5py.File(self.h5path, 'r')
        if splits == None:
            self.video_list = list(self.hf.keys())
        else:
            self.video_list = list(splits)
        self.with_images = with_images
        self.video_key_name = {}
        self.video_key_images = {}
        self.frame_dir = image_dir
        self.video_dir = video_dir
        self.transform = transform
        self.frame_source = None
        if with_images:
            assert image_dir != None or video_dir != None
            assert mapping_file_path != None
            if video_dir != None:
                self.frame_source = "video"
            else:
                self.frame_source = "image"
            with open(mapping_file_path) as fp:
                video_items = json.load(fp)
                for item in video_items:
                    video_file = item['name']
                    self.video_key_name[item['h5_key']] = ".".join(video_file.split(".")[:-1])
        if self.frame_source == "image":
            for video_key in self.video_list:
                video_name = self.video_key_name[video_key]
                video_frame_dir = os.path.join(self.frame_dir, video_name)
                frame_list = os.listdir(video_frame_dir)
                frame_items = list(map(lambda x: (VideoFeatureDataset._extract_file_id(x), x), frame_list))
                frame_items.sort(key=lambda x: x[0])
                frame_files = list(map(lambda x: x[1], frame_items))
                self.video_key_images[video_key] = frame_files
        elif self.frame_source == "video":
            pass

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_key = self.video_list[index]
        video_data = self.hf[video_key]['features']
        video_data = torch.Tensor(video_data)
        change_points = self.hf[video_key]['change_points'][...]
        num_frames = self.hf[video_key]['n_frames'][()]
        nfps = self.hf[video_key]['n_frame_per_seg'][...].tolist()
        positions = self.hf[video_key]['picks'][...]
        user_summary = self.hf[video_key]['user_summary'][...]
        gtscore = self.hf[video_key]['gtscore'][...]
        # target = torch.from_numpy(gtscore).unsqueeze(0)
        # seq = torch.from_numpy(self.hf[video_key]['features'][...]).unsqueeze(0)
        if self.frame_source == "image":
            images = []
            image_files = self.video_key_images[video_key]
            video_name = self.video_key_name[video_key]
            for image_file in image_files[::15]:
                image_path = os.path.join(self.frame_dir, video_name, image_file)
                img = default_loader(image_path)
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            images = images[:video_data.size(0)]
            images = torch.stack(images)
            # print(images.size(0), "=", video_data.size(0))
            return video_data, self.video_list[index], change_points, num_frames, nfps, positions, user_summary, gtscore,\
                  video_key, images
        elif self.frame_source == "video":
            video_name = self.video_key_name[video_key]
            video_path = os.path.join(self.video_dir, video_name + ".mp4")
            images, fps = read_video_sample(video_path, 15)
            return video_data, self.video_list[index], change_points, num_frames, nfps, positions, user_summary, gtscore, \
                   video_key, images
        else:
            image_tensor = []
        return video_data, self.video_list[index], change_points, num_frames, nfps, positions, user_summary, gtscore, video_key, []

    @staticmethod
    def _extract_file_id(filename):
        idx = filename[3:-4]
        idx = int(idx)
        return idx


def get_feature_loader(root, video_list=[], with_images=False,
                       image_dir=None, video_dir=None, mapping_file_path=None):
    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return DataLoader(VideoFeatureDataset(root, splits=video_list, transform=transform, with_images=with_images,
                                          image_dir=image_dir, video_dir=video_dir,
                                          mapping_file_path=mapping_file_path), batch_size=1 )


if __name__ == '__main__':
    pass
