# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import json
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from models.adversarial_summarization import Summarizer, Discriminator  # , apply_weight_norm
from datetime import datetime
import os
import socket
from utils.summary import generate_summary, evaluate_summary
from pathlib import Path
from utils.metric import Metric
from exceptions import InvalidEvaluationMethod, InExistentLossInEvaluation
from evaluation import EvaluationMetric
from .base_solver import SolverBase


class TestSolver(SolverBase):
    def __init__(self, config):
        super(TestSolver, self).__init__(config)
        self.input_size = config.input_size
        self.with_images = config.with_images
        # print(ckpt['args'])
        # self.config = ckpt['args']

    def build(self, linear_compress, summarizer, discriminator):
        self.linear_compress = linear_compress
        self.summarizer = summarizer
        self.discriminator = discriminator
        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator])

    def to(self, device):
        self.linear_compress = self.linear_compress.to(device)
        self.summarizer = self.summarizer.to(device)
        self.discriminator = self.discriminator.to(device)
        self.feature_extractor = self.feature_extractor.to(device)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def run(self, batch_i, batch):
        image_features = batch[0]
        image_features = image_features.view(-1, self.input_size)
        image_features_ = Variable(image_features).cuda()

        video_key = batch[-2][0]

        # ---- Train sLSTM, eLSTM ----#
        # if self.config.verbose:
        #     tqdm.write('\nTraining sLSTM and eLSTM...')

        # [seq_len, 1, hidden_size]
        feature_l2_losses = []
        if self.with_images:
            images = batch[-1]
        else:
            images = None
        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
        scores, h_mu, h_log_variance, generated_features, h_t = self.summarizer(
            original_features, z=None, images=images, video_key=video_key)
        return scores
