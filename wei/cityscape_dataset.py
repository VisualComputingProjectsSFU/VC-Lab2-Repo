import os
import random
import numpy as np
import torch.nn
from torch.utils.data import Dataset
import PIL
import PIL.PngImagePlugin
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import train
import bbox_helper


class CityScapeDataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.classes = ['car', 'cargroup', 'person', 'persongroup', 'traffic sign']
        self.cropping_threshold = 0.5

        # Initialize variables.
        self.imgWidth = None
        self.imgHeight = None
        self.crop_coordinate = None

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

        self.prior_bboxes = bbox_helper.generate_prior_bboxes(prior_layer_cfg=prior_layer_cfg)

        # Pre-process parameters, normalize: (I-self.mean)/self.std.
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes. Labels are include
        [car, traffic sign, person]. irrelevant objects are set to 0.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4).
        :return bbox_label: matched classification label, dim: (num_priors).
        """
        # Prepare configurations.
        item = self.dataset_list[idx]
        self.imgWidth = float(item['imgWidth'])
        self.imgHeight = float(item['imgHeight'])
        self.resize_ratio = min(self.imgHeight / 300., self.imgWidth / 300.)
        self.raw_labels = self.sanitize(item)

        # Resize the image and label first.
        file_path = os.path.join(train.data_path, item['file'])
        image = self.resize(Image.open(file_path))
        self.raw_labels = self.resize(self.raw_labels)

        # Prepare image array first to update crop.
        image = self.crop(image)
        image = self.flip(image)
        image = self.brighten(image)
        image = self.normalize(image)
        image = np.array(image)

        # Prepare labels second to apply crop.
        labels = self.crop(self.raw_labels)
        labels = self.flip(labels)
        labels = self.normalize(labels)
        labels = np.array(labels)

        # TODO: implement data loading
        # 1. Load image as well as the bounding box with its label
        # 2. Normalize the image with self.mean and self.std
        # 4. Normalize the bounding box position value from 0 to 1

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for i in range(labels.shape[0]):
            corner = bbox_helper.center2corner(labels[i, len(self.classes):])
            x = corner[0]
            y = corner[1]
            rect = patches.Rectangle((x, y), labels[i][len(self.classes) + 2], labels[i][len(self.classes) + 3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

        sample_labels = None
        sample_bboxes = None
        sample_img = None

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box
        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = bbox_helper.match_priors(self.prior_bboxes, sample_bboxes, sample_labels, iou_threshold=0.5)

        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return bbox_tensor, bbox_label_tensor

    def sanitize(self, item):
        labels = []
        for obj in item['objects']:
            label = np.zeros(len(self.classes))

            try:
                label[self.classes.index(obj['label'])] = 1.0
                polygons = np.array(obj['polygon'])
                corner = [min(polygons[:, 0]), min(polygons[:, 1]), max(polygons[:, 0]), max(polygons[:, 1])]

                # Add bounding box coordinates.
                labels.append(np.concatenate((label, bbox_helper.corner2center(corner))))

            except ValueError:
                pass
        return np.array(labels)

    def resize(self, inp):
        # Case for image input.
        if isinstance(inp, PIL.PngImagePlugin.PngImageFile):
            if self.imgWidth < self.imgHeight:
                self.imgWidth = 300
                self.imgHeight = self.imgHeight / self.resize_ratio
            else:
                self.imgWidth = self.imgWidth / self.resize_ratio
                self.imgHeight = 300
            return inp.resize((int(self.imgWidth), int(self.imgHeight)), Image.ANTIALIAS)

        # Case for label input.
        inp = np.concatenate(
            (inp[:, 0:len(self.classes)], np.true_divide(inp[:, len(self.classes):], self.resize_ratio)), axis=1)
        return inp

    def crop(self, inp):
        # Case for image input.
        if isinstance(inp, PIL.Image.Image):
            image = inp

            # Check the iosmall of the cropped image with oracle bounding box to ensure at least one labeled item.
            found = False
            while not found:
                crop = random.randint(0, self.imgWidth - 301)
                self.crop_coordinate = [crop, 0, crop + 300, 300]
                for label in self.raw_labels:
                    a = torch.Tensor([label[-4:]])
                    b = torch.Tensor([bbox_helper.corner2center(self.crop_coordinate)])

                    if bbox_helper.ios(a, b) > self.cropping_threshold:
                        found = True
                        image = image.crop(self.crop_coordinate)
                        break

            return np.asarray(image)

        # Case for label input.
        labels = inp
        labels[:, len(self.classes)] -= self.crop_coordinate[0]

        # Remove label with too small ios.
        ios = bbox_helper.ios(torch.Tensor(labels[:, len(self.classes):]), torch.Tensor([[150, 150, 300, 300]]))
        labels[np.where(ios <= self.cropping_threshold)] = 0
        labels = labels[~np.all(labels == 0, axis=1)]

        # Clip the label.
        for i_label in range(0, labels.shape[0]):
            corner = bbox_helper.center2corner(labels[i_label, len(self.classes):])
            corner[0] = max(corner[0], 0)
            corner[1] = min(corner[1], 300)
            corner[2] = max(corner[2], 0)
            corner[3] = min(corner[3], 300)
            center = bbox_helper.corner2center(corner)
            labels[i_label, len(self.classes):] = center

        return labels

    def flip(self, inp):
        return inp

    def brighten(self, inp):
        return inp

    def normalize(self, inp):
        # Case for image input.
        if inp.shape == (300, 300, 3):
            print(inp[0][0])
            image = inp
            image = np.subtract(image, self.mean)
            print(image[0][0])

            return image

        # Case for label input.
        labels = inp

        return labels

    def preview(self, index=-1):
        if index == -1:
            index = random.randint(0, len(self.dataset_list))

