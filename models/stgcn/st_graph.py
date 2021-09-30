# The code comes from
# https://github.com/jd730/STRG/blob/ef9d17d1b2dc129bae848e2d0f40a6e0731c7851/module/roi_graph.py#L27
import torch
import os
import pickle
import numpy as np
import torch
import torch.nn as n
import torch.nn.functional as F
import pdb


def get_iou(roi, rois, area, areas) :
    y_min = torch.max(roi[:,0:1], rois[:,:,0])
    x_min = torch.max(roi[:,1:2], rois[:,:,1])
    y_max = torch.min(roi[:,2:3], rois[:,:,2])
    x_max = torch.min(roi[:,3:4], rois[:,:,3])
    axis0 = x_max - x_min + 1
    axis1 = y_max - y_min + 1
    axis0[axis0 < 0] = 0
    axis1[axis1 < 0] = 0
    intersection = axis0 * axis1
    iou = intersection / (areas + area - intersection)
    return iou


def get_iou_without_batch(roi, rois, area, areas):
    y_min = torch.max(roi[0:1], rois[:, 0])
    x_min = torch.max(roi[1:2], rois[:, 1])
    y_max = torch.min(roi[2:3], rois[:, 2])
    x_max = torch.min(roi[3:4], rois[:, 3])
    axis0 = x_max - x_min + 1
    axis1 = y_max - y_min + 1
    axis0[axis0 < 0] = 0
    axis1[axis1 < 0] = 0
    intersection = axis0 * axis1
    iou = intersection / (areas + area - intersection)
    return iou


def get_spatial_graph(rois, threshold=0):
    # Inputs
    # - rois Tensor with size (n_object, 4)
    n_roi = rois.size(0)
    graph = torch.zeros(n_roi, n_roi)
    areas = (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 2] - rois[:, 0] + 1)
    for i in range(n_roi):
        ious = get_iou_without_batch(rois[i], rois, areas[i:i + 1], areas)
        ious[ious < threshold] = 0
        graph[i, :] = ious
    return graph

def get_st_graph(rois, threshold=0):
    B, T, N, _ = rois.size()

    M = T*N
    front_graph = torch.zeros((B,M,M))

    if M ==0 :
        return front_graph, front_graph.transpoes(1,2)
    areas = (rois[:,:,:,3] - rois[:,:,:,1] + 1) * (rois[:,:,:,2] - rois[:,:,:,0] + 1)

    for t in range(T-1):
        for i in range(N):
            print(rois[:, t, i].size())
            ious = get_iou(rois[:,t,i], rois[:,t+1], areas[:,t,i:i+1], areas[:,t+1])
            ious[ious < threshold] = 0
            front_graph[:, t*N+i, (t+1)*N:(t+2)*N] = ious

    back_graph = front_graph.transpose(1,2)

    # Normalize
    front_graph = front_graph / front_graph.sum(dim=-1, keepdim=True)
    back_graph = back_graph / back_graph.sum(dim=-1, keepdim=True)
    # NaN to zero
    front_graph[front_graph != front_graph] = 0
    back_graph[back_graph != back_graph] = 0

    return front_graph, back_graph
