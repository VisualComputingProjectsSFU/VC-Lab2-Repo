import numpy as np
import torch.nn
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors


class CityScapeDataset(Dataset):

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

        prior_layer_cfg = [
            {'layer_name': 'Layer11',
             'feature_dim_hw': (38, 38), 'bbox_size': (60, 60), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0)},
            {'layer_name': 'Layer23',
             'feature_dim_hw': (19, 19), 'bbox_size': (95, 95), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0)},
            {'layer_name': 'Layer27',
             'feature_dim_hw': (10, 10), 'bbox_size': (130, 130), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0)},
            {'layer_name': 'Layer29',
             'feature_dim_hw': (5, 5), 'bbox_size': (165, 165), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0)},
            {'layer_name': 'Layer31',
             'feature_dim_hw': (3, 3), 'bbox_size': (200, 200), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0)},
            {'layer_name': 'Layer33',
             'feature_dim_hw': (2, 2), 'bbox_size': (235, 235), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0)},
            {'layer_name': 'Layer35',
             'feature_dim_hw': (1, 1), 'bbox_size': (270, 270), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0)}
        ]

        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg=prior_layer_cfg)

        # Pre-process parameters, normalize: (I-self.mean)/self.std.
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4).
        :return bbox_label: matched classification label, dim: (num_priors).
        """

        # TODO: implement data loading
        # 1. Load image as well as the bounding box with its label
        # 2. Normalize the image with self.mean and self.std
        # 3. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        # 4. Normalize the bounding box position value from 0 to 1
        sample_labels = None
        sample_bboxes = None
        sample_img = None

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box

        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, sample_bboxes, sample_labels, iou_threshold=0.5)

        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return bbox_tensor, bbox_label_tensor
