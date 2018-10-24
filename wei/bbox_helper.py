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
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4).
    """

    # Configuration parameters.
    priors_bboxes = []
    m = len(prior_layer_cfg)

    for k in range(1, m + 1):
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
    priors_bboxes = torch.Tensor(priors_bboxes)
    priors_bboxes = torch.clamp(priors_bboxes, 0.0, 1.0)

    return priors_bboxes


def preprocess(a: torch.Tensor, b: torch.Tensor):
    """
    # Preprocess the tensor so both input will be in the same size.
    :param a: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :return: bounding box tensors with same dimension.
    """
    if a.shape == b.shape:
        return a, b
    if b.shape == torch.Size([4]):
        b = b.unsqueeze(0)
    b = b.repeat(a.shape[0], 1)

    return a, b


def intersect(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection.
    :param a: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :return: intersections values: dim: (n_item).
    """
    # Compute the intersections recangle.
    rec_a = center2corner(a)
    rec_b = center2corner(b)
    intersections = torch.cat((torch.max(rec_a, rec_b)[..., :2], torch.min(rec_a, rec_b)[..., 2:]), -1)

    # Compute the intersections area.
    x1 = intersections[..., 0]
    y1 = intersections[..., 1]
    x2 = intersections[..., 2]
    y2 = intersections[..., 3]
    sub1 = torch.sub(x2, x1)
    sub2 = torch.sub(y2, y1)
    sub1[sub1 < 0] = 0
    sub2[sub2 < 0] = 0

    intersections = torch.mul(sub1, sub2)

    return intersections


def iou(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection over Union.
    Note: function iou(a, b) used in match_priors.
    :param a: bounding boxes, dim: (n_items, 4).
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :return: iou value: dim: (n_item).
    """
    a, b = preprocess(a, b)
    intersections = intersect(a, b)
    area_a = torch.mul(a[..., 2], a[..., 3])
    area_b = torch.mul(b[..., 2], b[..., 3])

    return torch.div(intersections, torch.sub(torch.add(area_a, area_b), intersections))


def ios(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the intersection over smaller object.
    :param a: area, dim: (n_items, 4).
    :param b: area, dim: (n_items, 4).
    :return: intersection over smaller value: dim: (n_items).
    """
    a, b = preprocess(a, b)
    intersections = intersect(a, b)
    area_a = torch.mul(a[..., 2], a[..., 3])
    area_b = torch.mul(b[..., 2], b[..., 3])

    return torch.div(intersections, torch.min(area_a, area_b))


def match_priors(
        bounding_boxes: torch.Tensor,
        locations: torch.Tensor,
        iou_threshold: float):
    """
    Match the ground-truth boxes with the priors. Used in cityscape_dataset.py.
    :param bounding_boxes: bounding boxes, dim: (num_sample, 4).
    :param locations: location vector with attached bounding box, dim: (num_match, 4).
    :param iou_threshold: matching criterion.
    :return matched confidence labels.
    """
    matches = torch.zeros(bounding_boxes.shape[0])

    for i_box in range(bounding_boxes.shape[0]):
        iou_score = iou(locations, bounding_boxes[i_box])
        value, index = iou_score.max(0)
        if value > iou_threshold:
            value, index = locations[index].max(0)
            matches[i_box] = index + 1

    return matches


''' NMS ----------------------------------------------------------------------------------------------------------------
'''


def nms_bbox(confidences, locations, overlap_threshold=0.5, prob_threshold=0.6):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.

    :param locations: bounding box loc and size, dim: (num_priors, 4).
    :param confidences: bounding box confidence probabilities, dim: (num_priors, num_classes).
    :param overlap_threshold: the overlap threshold for filtering out outliers.
    :param prob_threshold: threshold to filter out boxes with low confidence level.
    :return: selected bounding box with classes.
    """

    # [DEBUG] Check if input is the desire shape.
    assert locations.dim() == 2
    assert locations.shape[1] == 4
    assert confidences.dim() == 2
    assert confidences.shape[0] == locations.shape[0]

    bboxes = []
    num_dim = locations.shape[1]
    num_class = confidences.shape[1]
    for i_bbox in range(0, locations.shape[0]):
        bboxes.append(torch.Tensor.numpy(locations[i_bbox]))
        bboxes.append(torch.Tensor.numpy(confidences[i_bbox]))

    bboxes = np.array(bboxes)
    bboxes = bboxes.reshape(locations.shape[0], -1)

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
    locations = torch.Tensor(bboxes[:, 0:4])
    confidences = torch.Tensor(bboxes[:, 4:])

    return confidences, locations


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
    return torch.cat([corner[..., 2:] / 2 + corner[..., :2] / 2,
                      corner[..., 2:] - corner[..., :2]], dim=-1)
