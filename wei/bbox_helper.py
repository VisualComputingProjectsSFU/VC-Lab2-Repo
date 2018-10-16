import torch
import numpy as np
import math
import cityscape_dataset

''' Prior Bounding Box  ------------------------------------------------------------------------------------------------
'''


# noinspection PyUnresolvedReferences
def generate_prior_bboxes(prior_layer_cfg):
    """
    Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

    According to formula S_k = S_min + (S_max - S_min)/(m - 1) * (k - 1)
    S_min = 0.2; S_max = 0.9; m = 7, # of feature maps will be regressed; k = current feature map between 1 and m.

    Use MobileNet_SSD 300x300 as example:
    Feature map dimension for each output layers:
       Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
    1. 11       | (38x38)        | (30x30) (unit. pixels)
    2. 23       | (19x19)        | (70x70)
    3. 27       | (10x10)        | (110x110)
    4. 29       | (5x5)          | (150x150)
    5. 31       | (3x3)          | (190x190)
    6. 33       | (2x2)          | (230x230)
    7. 35       | (1x1)          | (270x270)
    NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
    Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4).

    example_prior_layer_cfg = [
        {'layer_name': 'Layer11',
         'feature_dim_hw': (38, 38), 'bbox_size': (60, 60), 'aspect_ratio': (1.0, 1/2, 1/3, 2.0, 3.0)},
    ]
    """

    # Configuration parameters.
    priors_bboxes = []
    m = len(prior_layer_cfg)

    for k in range(1, m):
        # Read data from cfg.
        layer_cfg = prior_layer_cfg[k - 1]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']

        s_min = cityscape_dataset.CityScapeDataset.s_min
        s_max = cityscape_dataset.CityScapeDataset.s_max

        sk = s_min + (s_max - s_min)/(m - 1) * (k - 1)
        fk = layer_cfg['bbox_size'][0]

        # Compute center of the bounding boxes.
        for y in range(0, layer_feature_dim[0]):
            for x in range(0, layer_feature_dim[1]):
                cx = (x + 0.5)/fk
                cy = (y + 0.5)/fk

                # Add default box when ratio is 1.
                sk_next = s_min + (s_max - s_min) / (m - 1) * (k + 1 - 1)
                sk_mid = math.sqrt(sk * sk_next)

                h = sk_mid * math.sqrt(1)
                w = sk_mid / math.sqrt(1)
                priors_bboxes.append([cx, cy, w, h])

                # Compute the with and height of the bounding boxes.
                for aspect_ratio in layer_aspect_ratio:
                    h = sk * math.sqrt(aspect_ratio)
                    w = sk / math.sqrt(aspect_ratio)
                    priors_bboxes.append([cx, cy, w, h])

    # Convert to Tensor.
    priors_bboxes = torch.Tensor(priors_bboxes).cuda()
    priors_bboxes = torch.clamp(priors_bboxes, 0.0, 1.0)

    return priors_bboxes


def iou(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection over Union.
    Note: function iou(a, b) used in match_priors.
    :param a: bounding boxes, dim: (n_items, 4).
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :return: iou value: dim: (n_item).
    """
    # [DEBUG] Check if input is the desire shape.
    assert a.dim() == 2
    assert a.shape[1] == 4
    assert b.dim() == 2
    assert b.shape[1] == 4

    # Handle the case if b is a reference.
    if b.shape[0] == 1 and a.shape[0] > 1:
        b = b.repeat(a.shape[0], 1)

    # Decide the relationship and compute IoU.
    iou_list = []
    box1 = center2corner(a)
    box2 = center2corner(b)
    for index in range(0, a.shape[0]):

        # Compute the intersection.
        x1 = max(box1[index][0], box2[index][0])
        y1 = max(box1[index][1], box2[index][1])
        x2 = min(box1[index][2], box2[index][2])
        y2 = min(box1[index][3], box2[index][3])

        if x1 > x2 or y1 > y2:
            intersection = 0
        else:
            intersection = (x2 - x1) * (y2 - y1)

        # Compute the area for a and b.
        area_a = a[index][2] * a[index][3]
        area_b = b[index][2] * b[index][3]

        iou_list.append(intersection / (area_a + area_b - intersection))

    iou_tensor = torch.Tensor(iou_list).cuda()

    # [DEBUG] Check if output is the desire shape.
    assert iou_tensor.dim() == 1
    assert iou_tensor.shape[0] == a.shape[0]
    return iou_tensor


def ios(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the intersection over smaller object.
    :param a: area, dim: (n_items, 4).
    :param b: area, dim: (n_items, 4).
    :return: intersection over smaller value: dim: (n_items).
    """
    # Handle the case if b is a reference.
    if b.shape[0] == 1 and a.shape[0] > 1:
        ndb = np.array(b)
        ndb = ndb.repeat(a.shape[0], axis=0)
        b = torch.Tensor(ndb)

    # Decide the relationship and compute.
    ios_list = []
    box1 = center2corner(a)
    box2 = center2corner(b)
    for index in range(0, a.shape[0]):

        # Compute the intersection.
        x1 = max(box1[index][0], box2[index][0])
        y1 = max(box1[index][1], box2[index][1])
        x2 = min(box1[index][2], box2[index][2])
        y2 = min(box1[index][3], box2[index][3])

        if x1 > x2 or y1 > y2:
            intersection = 0
        else:
            intersection = (x2 - x1) * (y2 - y1)

        # Compute the area for a and b.
        area_a = a[index][2] * a[index][3]
        area_b = b[index][2] * b[index][3]

        ios_list.append(intersection / min(area_a, area_b))

    ios_tensor = torch.Tensor(ios_list).cuda()

    return ios_tensor


def match_priors(
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        iou_threshold: float):
    """
    Match the ground-truth boxes with the priors. Used in cityscape_dataset.py.
    :param bboxes: bounding boxes, dim: (num_sample, 4).
    :param labels: label vector with attached bounding box, dim: (num_match, num_class + 4).
    :param iou_threshold: matching criterion.
    :return labels: real matched labels mapped to all bounding boxes, dim: (num_sample, num_class + 4).
    """
    class_zeros = torch.zeros(bboxes.shape[0], labels.shape[1] - 4)

    for i_prior in range(bboxes.shape[0]):
        a = (bboxes[i_prior, -4:]).unsqueeze(0)
        b = labels[-4:]

        iou_score = iou(b, a)
        value, index = iou_score.max(0)

        if value > iou_threshold:
            class_zeros[i_prior] = labels[index, 0:-4]

    return torch.cat((class_zeros, bboxes), 1)


''' NMS ----------------------------------------------------------------------------------------------------------------
'''


def nms_bbox(bbox_locs, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.6):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.

    :param bbox_locs: bounding box loc and size, dim: (num_priors, 4).
    :param bbox_confid_scores: bounding box confidence probabilities, dim: (num_priors, num_classes).
    :param overlap_threshold: the overlap threshold for filtering out outliers.
    :param prob_threshold: threshold to filter out boxes with low confidence level.
    :return: selected bounding box with classes.
    """

    # [DEBUG] Check if input is the desire shape.
    assert bbox_locs.dim() == 2
    assert bbox_locs.shape[1] == 4
    assert bbox_confid_scores.dim() == 2
    assert bbox_confid_scores.shape[0] == bbox_locs.shape[0]

    bboxes = []
    num_dim = bbox_locs.shape[1]
    num_class = bbox_confid_scores.shape[1]
    for i_bbox in range(0, bbox_locs.shape[0]):
        bboxes.append(torch.Tensor.numpy(bbox_locs[i_bbox]))
        bboxes.append(torch.Tensor.numpy(bbox_confid_scores[i_bbox]))

    bboxes = np.array(bboxes)
    bboxes = bboxes.reshape(bbox_locs.shape[0], -1)

    # Preprocess to set any bounding box below confidence threshold to 0.
    for i_bbox in range(0, bboxes.shape[0]):
        if max(bboxes[i_bbox][4:]) < prob_threshold:
            bboxes[i_bbox] = np.zeros(num_dim + num_class)

    # Remove any row only contains zeros.
    bboxes = bboxes[~np.all(bboxes == 0, axis=1)]

    # NMS filter out unnecessary bounding boxes.
    for i_bbox in range(0, bboxes.shape[0]):
        if max(bboxes[i_bbox]) == 0:
            continue

        # Compare with other boxes.
        for j_bbox in range(0, bboxes.shape[0]):
            if max(bboxes[j_bbox]) == 0 or j_bbox == i_bbox:
                continue

            # Check if two boxes are for the same class.
            if np.argmax(bboxes[i_bbox][4:]) == np.argmax(bboxes[j_bbox][4:]):
                a = torch.Tensor([bboxes[i_bbox][0:4]])
                b = torch.Tensor([bboxes[j_bbox][0:4]])

                # Check if the two boxes overlap enough. And remove the lower confidence one.
                if iou(a, b)[0] > overlap_threshold:
                    if max(bboxes[j_bbox][4:]) > max(bboxes[i_bbox][4:]):
                        bboxes[i_bbox] = np.zeros(num_dim + num_class)
                    else:
                        bboxes[j_bbox] = np.zeros(num_dim + num_class)

    # Clean all removed boxes and convert to location and confidence.
    bboxes = bboxes[~np.all(bboxes == 0, axis=1)]
    bbox_locs = torch.Tensor(bboxes[:, 0:4])
    bbox_confid_scores = torch.Tensor(bboxes[:, 4:])

    return bbox_locs, bbox_confid_scores


''' Bounding Box Conversion --------------------------------------------------------------------------------------------
'''


def center2corner(center):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x, y) (x + w, y + h).
    :param center: bounding box in center form (cx, cy, w, h).
    :return: bounding box in corner form (x, y) (x + w, y + h).
    """
    return torch.cat([center[..., :2] - center[..., 2:] / 2,
                      center[..., :2] + center[..., 2:] / 2], dim=-1)


def corner2center(corner):
    """
    Convert bounding box from corner form (x, y) (x + w, y + h) to  center form (cx, cy, w, h).
    :param corner: bounding box in corner form (x, y) (x + w, y + h).
    :return: bounding box in center form (cx, cy, w, h).
    """
    return torch.cat([corner[..., 2:] + corner[..., :2] / 2,
                      corner[..., 2:] - corner[..., :2]], dim=-1)
