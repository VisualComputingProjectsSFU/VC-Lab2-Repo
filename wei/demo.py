import torch
import util.module_util
import ssd_net
import bbox_helper


example_prior_layer_cfg = [
    {'layer_name': 'Layer11',
     'feature_dim_hw': (38, 38), 'bbox_size': (60, 60), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0)},
    {'layer_name': 'Layer23',
     'feature_dim_hw': (19, 19), 'bbox_size': (95, 95), 'aspect_ratio': (1.0, 1 / 2, 2.0)},
    {'layer_name': 'Layer27',
     'feature_dim_hw': (10, 10), 'bbox_size': (130, 130), 'aspect_ratio': (1.0, 1 / 2, 2.0)},
    {'layer_name': 'Layer29',
     'feature_dim_hw': (5, 5), 'bbox_size': (165, 165), 'aspect_ratio': (1.0, 1 / 2, 2.0)},
    {'layer_name': 'Layer31',
     'feature_dim_hw': (3, 3), 'bbox_size': (200, 200), 'aspect_ratio': (1.0, 1 / 2, 2.0)},
    {'layer_name': 'Layer33',
     'feature_dim_hw': (2, 2), 'bbox_size': (235, 235), 'aspect_ratio': (1.0, 1 / 2, 2.0)},
    {'layer_name': 'Layer35',
     'feature_dim_hw': (1, 1), 'bbox_size': (270, 270), 'aspect_ratio': (1.0, 1 / 2, 2.0)}
]

# net = ssd_net.SsdNet(10)
# util.module_util.summary_layers(net, (3, 300, 300))

bbox = bbox_helper.generate_prior_bboxes(example_prior_layer_cfg)

box_a = torch.Tensor([
    [5., 5., 10., 10.],
    [5., 10., 10., 10.],
    [5., 10., 10., 10.],
    [50., 10., 10., 10.],
    [5., 5., 10., 10.],
])

box_b = torch.Tensor([
    [10., 10., 10, 10],
    [10., 5., 10, 10],
    [50., 10., 10., 10.],
    [50., 10., 10., 10.],
    [15., 5., 10., 10.],
])

print(bbox_helper.iou(box_a, box_b))
