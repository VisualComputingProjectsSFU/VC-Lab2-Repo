import ssd_net
import mobilenet
import bbox_helper
import torch
import torch.nn as nn
import torch.nn.functional as F
import module_util
import os
from collections import OrderedDict

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
#
# current_directory = os.getcwd()  # current working directory
#
# model = ssd_net.SSD(num_classes=4)
# func = module_util.summary_layers(model, (3, 300, 300))

# net = mobilenet.MobileNet()
#
#
# pre_trained_model = os.path.join(current_directory, 'pretrained/mobienetv2.pth')
#
# pret_state = torch.load(pre_trained_model)
# print(type(pret_state))
# print(pret_state.keys())
# print(net.state_dict().keys())
# print(len(pret_state))
# print(len(net.state_dict()))
# net.load_state_dict(pret_state)

# prior_layer_cfg = [{'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv13', 'feature_dim_hw': (10, 10), 'bbox_size': (102, 102),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (144, 144),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (186, 186),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (228, 228),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (270, 270),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
#                            ]
#
# prior_bboxes = bbox_helper.generate_prior_bboxes(prior_layer_cfg)