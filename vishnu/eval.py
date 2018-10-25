import ssd_net
import bbox_helper
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim
from torch.autograd import Variable
import matplotlib.patches as patches
import sys

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
current_directory = os.getcwd()  # current working directory

if __name__ == '__main__':

    img_file_path = sys.argv[1]

    test_img = Image.open(img_file_path)

    net = ssd_net.SSD(num_classes=6)

    model_path = os.path.join(current_directory, 'SSDnet_crop1.pth')

    net_state = torch.load(model_path)

    net.load_state_dict(net_state)

    net.cpu()

    net.eval()

    prior_layer_cfg = [{'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
                        'aspect_ratio': [2, 3, 4]},
                       {'layer_name': 'Conv13', 'feature_dim_hw': (10, 10), 'bbox_size': (102, 102),
                        'aspect_ratio': [2, 3, 4]},
                       {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (144, 144),
                        'aspect_ratio': [2, 3, 4]},
                       {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (186, 186),
                        'aspect_ratio': [2]},
                       {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (228, 228),
                        'aspect_ratio': [2]},
                       {'layer_name': 'Conv16_2', 'feature_dim_hw': (1, 1), 'bbox_size': (270, 270),
                        'aspect_ratio': [2]}
                       ]
    prior_bboxes = bbox_helper.generate_prior_bboxes(prior_layer_cfg)
    prior_bboxes = prior_bboxes.unsqueeze(0).cpu()



    test_img = np.array(test_img)
    test_img = np.subtract(test_img, [127, 127, 127])
    test_img = np.divide(test_img, 128)
    test_img = test_img.reshape((test_img.shape[2], test_img.shape[0], test_img.shape[1]))
    test_img = torch.Tensor(test_img)
    test_img = test_img.unsqueeze(0).cpu()

    images = Variable(test_img)  # Use Variable(*) to allow gradient flow\n",
    conf_preds, loc_preds = net.forward(images)  # Forward once\n",
    print(conf_preds.shape)
    print(loc_preds.shape)

    bbox = bbox_helper.loc2bbox(loc_preds, prior_bboxes)
    bbox = bbox[0].detach()
    bbox_corner = bbox_helper.center2corner(bbox)
    print(bbox_corner)
    print(conf_preds)
    print(bbox_corner.shape)
    bbox_corner = bbox_corner
    conf_preds = conf_preds[0].detach()

    # idx = conf_preds[:, 2] > 0.6
    # bbox_corner = bbox_corner[idx]
    # bbox = bbox[idx]
    # print(bbox_corner)

    bbox_nms = bbox_helper.nms_bbox(bbox_corner,conf_preds)
    print(bbox_nms[0])
    bbox_nms = torch.Tensor(bbox_nms[0])
    print(bbox_nms.shape)
    bbox_nms_cen = bbox_helper.corner2center(bbox_nms)

    test_img = test_img.detach()
    channels = test_img.shape[1]
    h, w = test_img.shape[2], test_img.shape[3]

    img_r = test_img.reshape(h, w, channels)
    img_n = (img_r + 1) / 2

    fig, ax = plt.subplots(1)

    ax.imshow(img_n)

    for index in range(0, bbox_nms.shape[0]):
        corner = bbox_nms[index]
        corner = torch.mul(corner, 300)
        x = corner[0]
        y = corner[1]
        raw_matched_rect = patches.Rectangle(
            (x, y),
            bbox_nms_cen[index, 2] * 300,
            bbox_nms_cen[index, 3] * 300,
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(raw_matched_rect)

    plt.show()

