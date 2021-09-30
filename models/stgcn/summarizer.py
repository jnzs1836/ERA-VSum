import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import pdb
import time

from .gcn import GCN, GraphConvolution
from .st_graph import get_st_graph
# from .models import RGCN

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model, GeneralizedRCNN
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from data.frame_loader import FrameData
from models.stgcn.st_gcn import RGCN
from models.adversarial_summarization.summarizer import VAE, sLSTM
import gc
import math
from cache import CacheBase
from .difference_attention import DifferenceAttention
from .gat import GraphAttentionLayer
def load_detectron_predictor():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)
    return predictor


class SpatioTemporalGraphBuilder(nn.Module):
    def __init__(self, in_channel=512, out_channel=512, test_mode=False,
                 dropout=0.5,
                 separate_fb=True):
        super(SpatioTemporalGraphBuilder, self).__init__()
        self.feature_channel = 256
        self.st_gcn = RGCN(self.feature_channel, self.feature_channel)
        self.detectron_predictor = load_detectron_predictor()
        self.max_pool = nn.MaxPool2d(14)
        self.last_seq_len = -1
        self.cache = {}

    def preprocess(self, frames):
        inputs = []
        frames = frames.squeeze()
        for i in range(frames.size(0)):
            frame = frames[i]
            # frame = frame.permute(1, 2, 0)
            frame = frame.detach().cpu().numpy()
            height, width = frame.shape[:2]
            image = self.detectron_predictor.aug.get_transform(frame).apply_image(frame)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": height, "width": width})
        return inputs

    def extract_rois_batch(self, inputs):
        batch_size = 64
        rois = []
        rois_features = []
        temporal_start_end = []
        current_t = 0
        with torch.no_grad():
            for i in range(math.ceil(len(inputs) / batch_size)):
                inputs_batch = inputs[i * batch_size: (i + 1) * batch_size]
                _rois, _rois_features, _temporal_start_end = self.extract_rois(inputs_batch)
                abs_temporal_start_end = []
                for start_end in _temporal_start_end:
                    abs_temporal_start_end.append([start_end[0] + current_t, start_end[1] + current_t])
                temporal_start_end.extend(abs_temporal_start_end)
                current_t = temporal_start_end[-1][-1]
                if _rois == None:
                    continue
                _rois = _rois.detach()
                _rois_features = _rois_features.detach()
                rois.append(_rois)
                rois_features.append(_rois_features)
                gc.collect()
                torch.cuda.empty_cache()

            if len(rois) == 0:
                return None, None, temporal_start_end
            rois = torch.cat(rois, dim=0)
            rois_features = torch.cat(rois_features)
            return rois, rois_features, temporal_start_end

    def extract_rois(self, inputs):
        model = self.detectron_predictor.model
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features)
        instances, _ = model.roi_heads(images, features, proposals)
        _mask_features = [features[f] for f in model.roi_heads.in_features]
        # mask_features = model.roi_heads.mask_pooler(_mask_features, [x.proposal_boxes[:15] for x in proposals])
        mask_features = model.roi_heads.mask_pooler(_mask_features, [x.pred_boxes for x in instances])
        if mask_features.size(0) == 0:
            return None, None, [[0, 0]] * len(inputs)
        rois_features = self.max_pool(mask_features).view(-1, self.feature_channel)
        # rois_features = rois_features.unsqueeze(0)
        # rois_features = rois_features.view(1, 4, 25, -1)
        temporal_start_end = []
        count = 0
        for instance in instances:
            t_instance_num = instance.pred_boxes.tensor.size(0)
            # t_instance_num = 15
            temporal_start_end.append([count, count + t_instance_num])
            count += t_instance_num
        rois = []
        # for proposal in proposals:
        #    rois.append(proposal.proposal_boxes[:15].tensor.detach())
        for instance in instances:
            rois.append(instance.pred_boxes.tensor.detach())
        rois = torch.cat(rois)

        return rois, rois_features, temporal_start_end

    def forward(self, frames):
        inputs = self.preprocess(frames)
        rois, rois_features, temporal_start_end = self.extract_rois_batch(inputs)
        
        return rois, rois_features, temporal_start_end

class SpatioTemporalFeatureExtractor(nn.Module):
    def __init__(self, in_channel=256, out_channel=512,
                 dropout=0.5, graph_model="gcn", stgcn_shortcut=False):
        super(SpatioTemporalFeatureExtractor, self).__init__()

        self.feature_channel = in_channel
        self.st_gcn = RGCN(self.feature_channel, self.feature_channel, dropout, graph_model, stgcn_shortcut=stgcn_shortcut)
        self.max_pool = nn.MaxPool2d(14)
        self.last_seq_len = -1
        self.cache = {}

    def forward(self, rois, rois_features, temporal_start_end):

        if rois == None:
            return torch.zeros(len(temporal_start_end), self.feature_channel).cuda()
        torch.cuda.empty_cache()
        rois = rois.detach()
        rois_features = rois_features.detach()
        gc.collect()
        torch.cuda.empty_cache()
        temporal_features = self.st_gcn(rois, rois_features, temporal_start_end)
        return temporal_features

class LSTMExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), 
            # nn.BatchNorm1d(hidden_size),# bidirection => scalar
            nn.ReLU())
        self.init_hidden()
        nn.init.kaiming_normal_(self.out[0].weight.data)
    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 100] (compressed pool5 features)
        Return:
            scores [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, 1]
        scores = self.out(features.squeeze(1))

        return scores
    
    def init_hidden(self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


class ScoringModule(nn.Module):
    def __init__(self, input_size, hidden_size, noise_dim, num_layers, supervised, graph_model, use_diff, stgcn_shortcut=False, dataset_type="summe"):
        super(ScoringModule, self).__init__()
        self.dataset_type = dataset_type
        self.s_lstm = LSTMExtractor(input_size + noise_dim, hidden_size, num_layers)
        self.graph_builder = CacheBase(SpatioTemporalGraphBuilder(), dataset_type)
        self.st_feature_extractor = SpatioTemporalFeatureExtractor(graph_model=graph_model, stgcn_shortcut=stgcn_shortcut)
        self.out_hidden_size = 128
        self.st_out = nn.Sequential(
            nn.Linear(256 + hidden_size, self.out_hidden_size),
            nn.BatchNorm1d(self.out_hidden_size),
            nn.ReLU(),
            nn.Linear(self.out_hidden_size, 1),
            nn.Sigmoid()
            # bidirection => scalar
            )
        nn.init.normal_(self.st_out[0].weight.data, 0, 1)
        nn.init.normal_(self.st_out[3].weight.data, 0, 1)
        # nn.init.kaiming_normal_(self.st_out[0].weight.data)
        self.supervised = supervised
        self.use_diff = use_diff
        if use_diff:
            self.difference_attention = DifferenceAttention(init_hidden=False)
        else:
            self.difference_attention = None

    def forward(self, image_features, z, images, video_key):
        st_graph = self.graph_builder.run(video_key, images)
        st_features = self.st_feature_extractor(*st_graph)
        image_features_noise = image_features

        if self.use_diff == True:
            diff_scores = self.difference_attention(image_features)
        else:
            diff_scores = None

        if z != None:
            image_features_noise = torch.cat([image_features, z], dim=2)
        # Apply weights
        gc.collect()
        torch.cuda.empty_cache()
        lstm_features = self.s_lstm(image_features_noise)
        features = torch.cat([lstm_features, st_features], dim=1)
        # print(features,"features")
        scores = self.st_out(features)
        if self.use_diff:
            scores = (scores + diff_scores) / 2
        return scores


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, noise_dim=0, num_layers=2, supervised=False, graph_model="gcn", use_diff=False, stgcn_shortcut=False, dataset_type="summe"):
        super().__init__()
        self.s_lstm = ScoringModule(input_size, hidden_size, noise_dim, num_layers, supervised, graph_model, use_diff, stgcn_shortcut=stgcn_shortcut, dataset_type=dataset_type)
        self.vae = VAE(input_size, hidden_size, num_layers)
        self.supervised = supervised
    def forward(self, image_features, uniform=False, z=None, images=None, video_key=None):
        # print("==+==")
        # print(self.st_out[0].weight.data)
        image_features_noise = image_features
        
        if not uniform:
            # [seq_len, 1]
                        # scores = (scores + st_scores)/ 2
            #if self.use_diff:
            #    scores = (scores + st_scores + diff_scores) / 3
            # [seq_len, 1, hidden_size]
            scores = self.s_lstm(image_features_noise, z, images, video_key)
            weighted_features = image_features * scores.view(-1, 1, 1)
        else:
            scores = None
            weighted_features = image_features
        if self.supervised:
            return scores.view(1, -1), None
        h_mu, h_log_variance, decoded_features, ht = self.vae(weighted_features)

        return scores, h_mu, h_log_variance, decoded_features, ht
