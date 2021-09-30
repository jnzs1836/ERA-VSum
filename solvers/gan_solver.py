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


class GANSolver(SolverBase):
    def __init__(self, config=None):
        super(GANSolver, self).__init__(config)
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.input_size = self.config.input_size
        self.evaluation_methods = config.evaluation_methods
        self.noise_dim = config.noise_dim
        self.best_k = config.best_k
        self.lambda_gp = 0.5
        self.lambda_l2 = 20
        self.feature_l2_loss_weight = 1
        self.with_images = config.with_images
        self.variance_type = config.variance_loss
        self.sparsity_type = config.sparsity_loss

        # self.resnet = ResNetFeature().cuda()

    def generate_noise(self, features_size):
        if self.noise_dim > 0:
            size = [features_size[0], features_size[1], features_size[2]]
            size[2] = self.noise_dim
            return torch.randn(size).cuda()
        else:
            return None

    def build(self, linear_compress, summarizer, discriminator):
        # Build Modules
        self.linear_compress = linear_compress
        self.summarizer = summarizer
        self.discriminator = discriminator

        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator])
        self.variance_loss_lambda = 10
        self.sparsity_loss_lambda = 10
        if self.config.mode == 'train':
            # Build Optimizers
            self.s_e_optimizer = optim.Adam(
                list(self.summarizer.s_lstm.parameters()) # + list(self.summarizer.st_feature_extractor.parameters()) + list(self.summarizer.st_out.parameters())
                + list(self.summarizer.vae.e_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr, weight_decay=self.config.weight_decay)
            self.d_optimizer = optim.Adam(
                list(self.summarizer.vae.d_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr, weight_decay=self.config.weight_decay)
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr, weight_decay=self.config.discriminator_weight_decay)
            self.s_e_scheduler = optim.lr_scheduler.StepLR(self.s_e_optimizer,
                                                           step_size=self.config.scheduler_step*20,
                                                           gamma=self.config.scheduler_gamma)
            self.d_scheduler = optim.lr_scheduler.StepLR(self.d_optimizer,
                                                         step_size=self.config.scheduler_step*20,
                                                         gamma=self.config.scheduler_gamma)
            self.c_scheduler = optim.lr_scheduler.StepLR(self.c_optimizer,
                                                         step_size=self.config.discriminator_scheduler_step*20,
                                                         gamma=self.config.discriminator_scheduler_gamma)


            self.model.train()



    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    def reconstruction_loss(self, h_origin, h_fake):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        return torch.norm(h_origin - h_fake, p=2)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(scores) - self.config.summary_rate)

    def dpp_sparsity_loss(self, seq, scores):
        seq = seq.squeeze()
        normed_seq = seq / seq.norm(p=2, dim=1, keepdim=True)
        scored_seq = normed_seq * scores
        sim_mat = torch.matmul(normed_seq, normed_seq.t())
        scored_sim_mat = torch.matmul(scored_seq, scored_seq.t())
        identity = torch.torch.eye(sim_mat.size(0)).cuda()
        a = torch.det(scored_sim_mat + identity)
        b = torch.det(sim_mat + identity)
        loss = torch.log(a/ b)
        return -loss

    def gan_loss(self, original_prob, fake_prob, uniform_prob):
        """Typical GAN loss + Classify uniformly scored features"""

        gan_loss = torch.mean(torch.log(original_prob) + torch.log(1 - fake_prob)
                              + torch.log(1 - uniform_prob))  # Discriminate uniform score

        return gan_loss
    def variance_loss(self, scores, epsilon=1e-4):
        median_tensor = torch.zeros(scores.shape[0]).to(self.device)
        median_tensor.fill_(torch.median(scores))
        loss = nn.MSELoss()
        variance = loss(scores.squeeze(), median_tensor)
        return 1 / (variance + epsilon)

    def normal_variance_loss(self, scores):
        scores = scores.squeeze()
        scores = scores * 2 - 1
        mu = torch.mean(scores)
        log_variance = torch.log(torch.var(scores))
        loss = 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)
        return loss

    def mean_median_variance(self, scores):
        scores = scores.squeeze()
        median = torch.median(scores)
        mean = torch.mean(scores)
        print(median - mean)
        return median - mean

    def score_sum_variance(self, scores):
        return scores.squeeze().sum() / np.sqrt(scores.squeeze().size(0))

    def score_target_variance(self, scores):
        return torch.norm(scores.squeeze().sum() - 0.15 * scores.squeeze().size(0),p=2) / np.sqrt(scores.squeeze().size(0))

    def my_variance_loss(self, scores):
        if self.variance_type == "median":
            return self.variance_loss(scores)
        elif self.variance_type == "starget":
            return self.variance_loss_lambda  * self.score_target_variance(scores)
        elif self.variance_type == "ssum":
            return self.variance_loss_lambda * self.score_sum_variance(scores)
        elif self.variance_type == "normal":
            return self.variance_loss_lambda * self.normal_variance_loss(scores)
        elif self.variance_type == "mean_median":
            return self.variance_loss_lambda * self.mean_median_variance(scores)
        else:
            return 0

    def my_sparsity_loss(self, seq, scores):
        if self.sparsity_type == "dpp":
            return self.sparsity_loss_lambda * self.dpp_sparsity_loss(seq, scores)
        elif self.sparsity_type == "slen":
            return self.sparsity_loss_lambda * self.sparsity_loss(scores)
        else:
            return 0
    def step(self, batch_i, batch, step_type="train"):
        image_features = batch[0]
        image_features = image_features.view(-1, self.input_size)
        image_features_ = Variable(image_features).cuda()
        # ---- Train sLSTM, eLSTM ----#
        # if self.config.verbose:
        #     tqdm.write('\nTraining sLSTM and eLSTM...')
        video_key = batch[-2][0]
        # [seq_len, 1, hidden_size]
        if self.with_images:
            images = batch[-1]
        else:
            images = None
        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
        scores, h_mu, h_log_variance, generated_features, h_t = self.summarizer(
            original_features, z=None, images=images, video_key=video_key)
        _, _, _, uniform_features, _ = self.summarizer(
            original_features, uniform=True, images=images, video_key=video_key)
        h_origin, original_prob = self.discriminator(original_features)
        h_fake, fake_prob = self.discriminator(generated_features)
        h_uniform, uniform_prob = self.discriminator(uniform_features)

        reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
        prior_loss = self.prior_loss(h_mu, h_log_variance)
        # sparsity_loss = self.sparsity_loss(scores)
        # tqdm.write(
        #     f'recon loss {reconstruction_loss.item():.3f}, prior loss: {prior_loss.item():.3f}, sparsity loss: {sparsity_loss.data.item():.3f}')
        sparsity_loss = self.my_sparsity_loss(h_t, scores)
        # dpp_sparsity_loss = self.dpp_sparsity_loss(h_t, scores)
        variance_loss = self.my_variance_loss(scores)
        s_e_loss = reconstruction_loss + prior_loss + sparsity_loss + variance_loss

        if step_type == "train":
            self.s_e_optimizer.zero_grad()
            s_e_loss.backward()  # retain_graph=True)
            # Gradient cliping
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
            self.s_e_optimizer.step()


        # ---- Train dLSTM ----#
        # if self.config.verbose:
        #     tqdm.write('Training dLSTM...')

        # [seq_len, 1, hidden_size]
        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

        scores, h_mu, h_log_variance, generated_features, h_t = self.summarizer(
            original_features, z=None, images=images, video_key=video_key)
        _, _, _, uniform_features, _ = self.summarizer(
            original_features, uniform=True, images=images, video_key=video_key)

        h_origin, original_prob = self.discriminator(original_features)
        h_fake, fake_prob = self.discriminator(generated_features)
        h_uniform, uniform_prob = self.discriminator(uniform_features)


        reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
        gan_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)


        d_loss = reconstruction_loss + gan_loss

        if step_type == "train":
            self.d_optimizer.zero_grad()
            d_loss.backward()  # retain_graph=True)
            # Gradient cliping
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
            self.d_optimizer.step()

        c_loss = 0
        # ---- Train cLSTM ----#
        if batch_i > self.config.discriminator_slow_start:
            # [seq_len, 1, hidden_size]
            original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

            scores, h_mu, h_log_variance, generated_features, h_t = self.summarizer(
                original_features, z=None, images=images, video_key=video_key)
            _, _, _, uniform_features, _ = self.summarizer(
                original_features, uniform=True, images=images, video_key=video_key)

            h_origin, original_prob = self.discriminator(original_features)
            h_fake, fake_prob = self.discriminator(generated_features)
            h_uniform, uniform_prob = self.discriminator(uniform_features)
           # Maximization
            c_loss = -1 * self.gan_loss(original_prob, fake_prob, uniform_prob)
            if step_type == "train":
                self.c_optimizer.zero_grad()
                c_loss.backward()
                # Gradient cliping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                self.c_optimizer.step()
        probs = {
            "original_prob": original_prob.data,
            "fake_prob": fake_prob.data,
            "uniform_prob": uniform_prob
        }
        losses = {
            "sparsity_loss": sparsity_loss.data,
            "variance_loss": variance_loss,
            "prior_loss": prior_loss.data,
            "gan_loss": gan_loss.data,
            "recon_loss": reconstruction_loss.data,
            "s_e_loss": s_e_loss.data,
            "d_loss": d_loss.data,
            "c_loss": c_loss
        }
        metrics = self.evaluate_results(scores, losses, probs, batch)
        return scores, losses, probs, metrics




    def pretrain(self):
        pass

    def get_model(self):
        return self.model


if __name__ == '__main__':
    pass
