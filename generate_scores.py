import sys
sys.path.append("../")
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import googlenet
from torchvision import transforms

import math
# from data.feature_loader import get_feature_loader
# from configs import get_config
# from solver import Solver
from data import get_frame_loader
from models.adversarial_summarization import Summarizer, Discriminator
from tqdm import tqdm
from pathlib import Path
from deployment.io import read_video, read_scores, read_write_selected_video_frames, read_video_meta
from deployment.postprocess import extract_boolean_frames, map_to_original_boolean_frames
# from generate_summary import sample_video_frames, DeploymentSolver, generate_scores
from factory.solver_factory import build_test_solver
from data import get_feature_loader
from configs import TestConfig
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="avs")
parser.add_argument('--model_name', type=str, default='test')
parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument("--output_dir", type=str,
                    default='')
parser.add_argument("--split_index", type=int, default=0)
args = parser.parse_args()


def main(args):
    ckpt = torch.load(args.ckpt_path)
    config = TestConfig(ckpt['args'])
    test_loader = get_feature_loader(config.video_path, config.splits[args.split_index]['test_keys'], config.with_images,
                                     config.image_dir, config.video_dir,
                                     mapping_file_path=config.mapping_file)
    test_solver = build_test_solver(config=config)
    test_solver.load_state_dict(ckpt['best_state_dict'])
    all_scores = {}
    for batch_i, batch in enumerate(tqdm(
        test_loader, desc='Batch', ncols=80, leave=False)):
        # for i in range(len(batch)):
        #     if torch.is_tensor(batch[i]):
        #         batch[i] = batch[i].cuda()
        batch[0] = batch[0].cuda()
        video_key = batch[-2][0]
        scores = test_solver.run(batch_i, batch)
        scores = scores.squeeze().cpu().detach().tolist()
        all_scores[video_key] = scores
    with open(os.path.join(args.output_dir, "{}.json".format(args.model_name)), "w") as fp:
        json.dump(all_scores, fp)


if __name__ == '__main__':
    main(args)

