import torch
import torch.nn as nn
import numpy as np
from evaluation import EvaluationMetric
from .base_solver import SolverBase

class SupervisedSolver(SolverBase):
    def __init__(self, config):
        self.config = config
        super(SupervisedSolver, self).__init__(config)
        self.evaluation_methods = config.evaluation_methods
        self.compressing_features = config.compressing_features
        self.variance_type = config.variance_loss


    def build(self, summarizer=None, linear_compress=None):
        self.summeaizer = summarizer.cuda()
        parameters = list(filter(lambda p: p.requires_grad, self.summeaizer.parameters()))
        if linear_compress != None:
            _parameteters = list(linear_compress.parameters())
            parameters = parameters + _parameteters
        self.optimizer = torch.optim.Adam(parameters, lr=self.config.lr, weight_decay=0.5)
        self.linear_compress = linear_compress
        self.model = nn.ModuleList([self.summeaizer, self.linear_compress])

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

    def score_sum_variance(self, scores):
        return scores.squeeze().sum() / np.sqrt(scores.squeeze().size(0))

    def score_target_variance(self, scores):
        return torch.norm(scores.squeeze().sum() - 0.15 * scores.squeeze().size(0),p=2) / np.sqrt(scores.squeeze().size(0))

    def my_variance_loss(self, scores):
        if self.variance_type == "median":
            return self.variance_loss(scores)
        elif self.variance_type == "starget":
            return 10 * self.score_target_variance(scores)
        elif self.variance_type == "ssum":
            return 10 * self.score_sum_variance(scores)
        elif self.variance_type == "normal":
            return 10 * self.normal_variance_loss(scores)
        else:
            return 0

    def step(self, batch_i, batch, step_type="train"):
        criterion = nn.MSELoss()
        image_features = batch[0].cuda()
        video_key = batch[-2][0]
        images = batch[-1].cuda()
        target = batch[7]
        target = target.cuda()
        # target = dataset['gtscore'][...]
        # target = torch.from_numpy(target).unsqueeze(0)
        # target = target.squeeze(0)
        # Normalize frame scores
        target -= target.min()
        target /= target.max()
        image_features = image_features.view(image_features.size(1), image_features.size(2))
        if self.compressing_features:
            image_features  = self.linear_compress(image_features.detach())
        image_features = image_features.unsqueeze(1)
        # seq_len = image_features.shape[1]
        y, _ = self.summeaizer(image_features, images=images, video_key=video_key)
        scores = y.view(-1)
        loss_att = 0

        loss = criterion(y, target)
        # loss2 = y.sum()/seq_len
        loss = loss + loss_att
        if step_type == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        losses = {
            "mse_loss": loss
        }
        probs = {}
        metrics = self.evaluate_results(scores, losses, probs, batch)
        return scores, losses, probs, metrics

