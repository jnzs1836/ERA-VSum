import torch
import torch.nn as nn
from .discriminator import cLSTM


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, mlp_hidden_size=128, use_patch=True):
        """Discriminator: cLSTM + output projection to probability"""
        super().__init__()
        self.cLSTM = cLSTM(input_size, hidden_size, num_layers, output_seq=True)
        self.use_patch = use_patch
        self.out = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1)
            )
        self.patch_conv = nn.Sequential(
            nn.Conv1d(hidden_size, mlp_hidden_size, 5, 1),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Conv1d(mlp_hidden_size, mlp_hidden_size * 2, 5, 2),
            nn.BatchNorm1d(mlp_hidden_size * 2),
            nn.ReLU(),
            nn.Conv1d(mlp_hidden_size*2, mlp_hidden_size, 1, 1),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Conv1d(mlp_hidden_size, mlp_hidden_size * 2, 5, 2),
            nn.BatchNorm1d(mlp_hidden_size * 2),
            nn.ReLU(),
            nn.Conv1d(mlp_hidden_size * 2, mlp_hidden_size, 1, 1),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(5, 2),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Conv1d(mlp_hidden_size, 1, 1, 1)
        )
    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h : [1, hidden_size]
                Last h from top layer of discriminator
            prob: [1=batch_size, 1]
                Probability to be original feature from CNN
        """

        # [1, hidden_size]
        h, seq_h = self.cLSTM(features)
        # patch_probs =
        # [1]
        if self.use_patch:
            seq_h = seq_h.permute(1, 2, 0)
            patch_prob = self.patch_conv(seq_h)
        else:
            patch_prob = self.out(seq_h)
        prob = patch_prob.squeeze()

        return h, prob


if __name__ == '__main__':

    pass
