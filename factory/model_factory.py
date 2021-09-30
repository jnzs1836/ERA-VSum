import torch
import torch.nn as nn
from models.adversarial_summarization import Discriminator, Summarizer, Critic
from models.stgcn.summarizer import Summarizer as SpatioTemporalSummarizer
from exceptions import InvalidModelException




def build_sum_gan_summarizer(config):
    use_diff = False
    if config.summarizer == "SUM-GAN-Diff":
        use_diff = True
    return Summarizer(
        input_size=config.hidden_size,
        noise_dim=config.noise_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers, use_diff=use_diff).cuda()




def build_st_graph_summarizer(config, supervised=False):
    graph_model = "gcn"
    if config.summarizer == "ST-GRAPH-GAT":
        graph_model = "gat"

    use_diff = False
    if config.summarizer == "ST-GRAPH-GAT-Diff" or config.summarizer == "ST-GRAPH-GCN-Diff":
        use_diff = True
    stgcn_shortcut = False
    if config.stgcn_shortcut == 1:
        stgcn_shortcut = True
    return SpatioTemporalSummarizer(
        input_size=config.hidden_size,
        noise_dim=config.noise_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        supervised=supervised,
        graph_model=graph_model,
        use_diff=use_diff,
        stgcn_shortcut=stgcn_shortcut,
        dataset_type=config.dataset_type
    ).cuda()

def build_sum_gan_compressor(config):
    return nn.Linear(
        config.input_size,
        config.hidden_size).cuda()


def build_sum_gan_discriminator(config):
    return Discriminator(
        input_size=config.hidden_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers).cuda()



def build_summarizer(config, supervised=False):
    if config.summarizer.startswith ("SUM-GAN"):
        return build_sum_gan_summarizer(config)
    elif config.summarizer == "ST-GRAPH-GAN" or config.summarizer.startswith("ST-GRAPH"):
        return build_st_graph_summarizer(config, supervised)
    else:
        raise InvalidModelException(config.summarizer)


def build_discriminator(config):
    if config.discriminator == "SUM-GAN":
        return build_sum_gan_discriminator(config)
    else:
        raise InvalidModelException(config.discriminator)


def build_sum_gan_critic(config):
    use_patch = True
    if config.critic == "SUM-GAN-NoPatch":
        use_patch = False
    return Critic(
        input_size=config.hidden_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        use_patch = use_patch
    ).cuda()


def build_critic(config):
    if config.critic.startswith("SUM-GAN"):
        return build_sum_gan_critic(config)
    else:
        raise InvalidModelException(config.critic)

def build_compressor(config):
    if config.compressor == "SUM-GAN":
        return build_sum_gan_compressor(config)
    else:
        raise InvalidModelException(config.compressor)
