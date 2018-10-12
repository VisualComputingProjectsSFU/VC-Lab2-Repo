import torch
import math

''' Prior Bounding Box  ------------------------------------------------------------------------------------------------
'''


def generate_prior_bboxes(prior_layer_cfg):
    """
    Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

    According to formula S_k = S_min + (S_max - S_min)/(m - 1) * (k - 1)
    S_min = 0.2; S_max = 0.9; m = 7, # of feature maps will be regressed; k = current feature map between 1 and m.

    Use MobileNet_SSD 300x300 as example:
    Feature map dimension for each output layers:
       Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
    1. 11       | (38x38)        | (60x60) (unit. pixels)
    2. 23       | (19x19)        | (95x95)
    3. 27       | (10x10)        | (130x130)
    4. 29       | (5x5)          | (165x165)
    5. 31       | (3x3)          | (200x200)
    6. 33       | (2x2)          | (235x235)
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
    s_min = 0.2
    s_max = 0.9
    m = len(prior_layer_cfg)

    for k in range(1, m):
        # Read data from cfg.
        layer_cfg = prior_layer_cfg[k - 1]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']

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

    # [DEBUG] Check the output shape.
    assert priors_bboxes.dim() == 2
    assert priors_bboxes.shape[1] == 4
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
    if b.shape[0] == 1 & a.shape[0] > 1:
        b = b.repeat(a.shape[0])

    # Decide the relationship and compute IoU.
    iou_list = []
    for index in range(0, a.shape[0]):
        if a[index][0] < b[index][0]:
            left = a[index]
            right = b[index]
        else:
            left = b[index]
            right = a[index]

        # Compute the intersection area.
        if left[1] > right[1]:
            nw_x = right[0] - right[2] / 2
            nw_y = right[1] + right[3] / 2
            se_x = left[0] + left[2] / 2
            se_y = left[1] - left[3] / 2
            intersection = max(se_x - nw_x, 0) * max(nw_y - se_y, 0)
        else:
            ne_x = left[0] + left[2] / 2
            ne_y = left[1] + left[3] / 2
            sw_x = right[0] - right[2] / 2
            sw_y = right[1] - right[3] / 2
            intersection = max(ne_x - sw_x, 0) * max(ne_y - sw_y, 0)

        # Compute the area for a and b.
        area_a = a[index][2] * a[index][3]
        area_b = b[index][2] * b[index][3]

        iou_list.append(intersection / (area_a + area_b - intersection))

    iou_tensor = torch.Tensor(iou_list)

    # [DEBUG] Check if output is the desire shape.
    assert iou_tensor.dim() == 1
    assert iou_tensor.shape[0] == a.shape[0]
    return iou_tensor


def match_priors(
        prior_bboxes: torch.Tensor,
        oracle_bboxes: torch.Tensor,
        oracle_labels: torch.Tensor,
        iou_threshold: float):
    """
    Match the ground-truth boxes with the priors. Used in cityscape_dataset.py.

    :param oracle_bboxes: ground-truth bounding boxes, dim:(n_samples, 4).
    :param oracle_labels: ground-truth classification labels, negative (background) = 0, dim: (n_samples).
    :param prior_bboxes: prior bounding boxes on different levels, dim:(num_priors, 4).
    :param iou_threshold: matching criterion.
    :return matched_boxes: real matched bounding box, dim: (num_priors, 4).
    :return matched_labels: real matched classification label, dim: (num_priors).
    """
    # [DEBUG] Check if input is the desire shape
    assert oracle_bboxes.dim() == 2
    assert oracle_bboxes.shape[1] == 4
    assert oracle_labels.dim() == 1
    assert oracle_labels.shape[0] == oracle_bboxes.shape[0]
    assert prior_bboxes.dim() == 2
    assert prior_bboxes.shape[1] == 4

    matched_boxes = None
    matched_labels = None

    # TODO: implement prior matching

    # [DEBUG] Check if output is the desire shape
    assert matched_boxes.dim() == 2
    assert matched_boxes.shape[1] == 4
    assert matched_labels.dim() == 1
    assert matched_labels.shape[0] == matched_boxes.shape[0]

    return matched_boxes, matched_labels


''' NMS ----------------------------------------------------------------------------------------------------------------
'''


def nms_bbox(bbox_loc, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.6):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.

    :param bbox_loc: bounding box loc and size, dim: (num_priors, 4).
    :param bbox_confid_scores: bounding box confidence probabilities, dim: (num_priors, num_classes).
    :param overlap_threshold: the overlap threshold for filtering out outliers.
    :return: selected bounding box with classes.
    """

    # [DEBUG] Check if input is the desire shape
    assert bbox_loc.dim() == 2
    assert bbox_loc.shape[1] == 4
    assert bbox_confid_scores.dim() == 2
    assert bbox_confid_scores.shape[0] == bbox_loc.shape[0]

    sel_bbox = []

    # Todo: implement nms for filtering out the unnecessary bounding boxes
    num_classes = bbox_confid_scores.shape[1]
    for class_idx in range(0, num_classes):

        # Tip: use prob_threshold to set the prior that has higher scores and filter out the low score items for fast
        # computation

        pass

    return sel_bbox


''' Bounding Box Conversion --------------------------------------------------------------------------------------------
'''


def loc2bbox(loc, priors, center_var=0.1, size_var=0.2):
    """
    Compute SSD predicted locations to boxes(cx, cy, h, w).
    :param loc: predicted location, dim: (N, num_priors, 4).
    :param priors: default prior boxes, dim: (1, num_prior, 4).
    :param center_var: scale variance of the bounding box center point.
    :param size_var: scale variance of the bounding box size.
    :return: boxes: (cx, cy, h, w).
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # Prior bounding boxes.
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # Locations.
    l_center = loc[..., :2]
    l_size = loc[..., 2:]

    # Real bounding box.
    return torch.cat([
        center_var * l_center * p_size + p_center,      # b_{center}
        p_size * torch.exp(size_var * l_size)           # b_{size}
    ], dim=-1)


def bbox2loc(bbox, priors, center_var=0.1, size_var=0.2):
    """
    Compute boxes (cx, cy, h, w) to SSD locations form.
    :param bbox: bounding box (cx, cy, h, w) , dim: (N, num_priors, 4).
    :param priors: default prior boxes, dim: (1, num_prior, 4).
    :param center_var: scale variance of the bounding box center point.
    :param size_var: scale variance of the bounding box size.
    :return: loc: (cx, cy, h, w).
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # Prior bounding boxes.
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # Locations.
    b_center = bbox[..., :2]
    b_size = bbox[..., 2:]

    return torch.cat([
        1 / center_var * ((b_center - p_center) / p_size),
        torch.log(b_size / p_size) / size_var
    ], dim=-1)


def center2corner(center):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h).
    :param center: bounding box in center form (cx, cy, w, h).
    :return: bounding box in corner form (x,y) (x+w, y+h).
    """
    return torch.cat([center[..., :2] - center[..., 2:]/2,
                      center[..., :2] + center[..., 2:]/2], dim=-1)


def corner2center(corner):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h).
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([corner[..., :2] - corner[..., 2:]/2,
                      corner[..., :2] + corner[..., 2:]/2], dim=-1)