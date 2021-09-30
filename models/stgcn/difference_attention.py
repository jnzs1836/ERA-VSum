# -*- coding: utf-8 -*-
import h5py
import numpy as np
import torch
import torch.nn as nn
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DifferenceModule(nn.Module):
    def __init__(self, input_size=500, diff=0):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        super().__init__()
        self.input_size = input_size
        self.diff = diff
        self.fc = nn.Linear(self.input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, features):
        diff = self.diff
        features = features.squeeze()
        features_diff = torch.zeros_like(features)
        features_diff[:features.size(0) - diff] = features[diff:]
        features_diff[features.size(0) - diff:] = features[-1]
        features_diff = features_diff.detach()
        h = torch.abs(features_diff - features)
        h = self.fc(h)
        h = self.relu(h)

        return h


class DifferenceAttention(nn.Module):
    def __init__(self, input_size=500, init_hidden=False):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        super().__init__()
        self.input_size = input_size
        self.diff_2 = DifferenceModule(input_size, 2)
        self.diff_4 = DifferenceModule(input_size, 4)
        self.diff_1 = DifferenceModule(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.model = nn.ModuleList([self.diff_1, self.diff_2, self.diff_4])
        self.init_hidden = init_hidden
        if self.init_hidden:
            self.init_hidden()

    def init_hidden(self):
        nn.init.kaiming_normal_(self.model.weight.data)



    def forward(self, image_features):
        diff_1 = self.diff_1(image_features)
        diff_2 = self.diff_2(image_features)
        diff_4 = self.diff_4(image_features)
        diff = diff_1 + diff_2 + diff_4
        diff = self.sigmoid(diff)
        return diff
