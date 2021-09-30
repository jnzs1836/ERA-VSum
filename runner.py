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
from solvers import GANSolver
from loggers import Logger, DebugLogger
from factory import build_solver

class Runner(object):
    def __init__(self, config=None, train_loader=None, test_loader=None, split_id=0):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = test_loader
        self.test_loader = test_loader
        self.split_id = split_id
        # self.resnet = ResNetFeature().cuda()
        self.solver = build_solver(config)

    def build(self):
        self.run_name = self.config.run_name + "_" + "split-" + str(self.split_id)
        self.save_dir = self.config.run_save_dir.joinpath("split-" + str(self.split_id))
        os.mkdir(self.save_dir)
        self.score_dir = self.config.run_score_dir.joinpath("split-" + str(self.split_id))
        os.mkdir(self.score_dir)
        if self.config.debug:
            self.logger = DebugLogger(os.path.join(self.config.run_log_dir, "split-" + str(self.split_id)), self.save_dir,
                             self.config, self.config.optimal_criteria)
        else:
            self.logger = Logger(os.path.join(self.config.run_log_dir, "split-" + str(self.split_id)), self.save_dir,
                             self.config, self.config.optimal_criteria)

    def train(self):
        step = 0
        valid_step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            for batch_i, batch in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):
                # for i in range(len(batch)):
                #     if torch.is_tensor(batch[i]):
                #         batch[i] = batch[i].cuda()
                batch[0] = batch[0].cuda()
                torch.cuda.empty_cache()
                scores, losses, probs, metrics = self.solver.train_step(batch_i, batch)
                self.logger.log_train_step(step, scores, losses, probs, metrics)
                step += 1
                # break
            self.logger.log_train_epoch(epoch_i)

            for batch_i, batch in enumerate(tqdm(self.test_loader, desc="validation")):
                scores, losses, probs, metrics = self.solver.valid_step(batch_i, batch)
                self.logger.log_valid_step(valid_step, scores, losses, probs, metrics)
                valid_step += 1
                print(metrics)
                # break
            self.logger.log_valid_epoch(epoch_i, self.solver.get_model())

    def pretrain(self):
        pass


if __name__ == '__main__':
    pass
