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
    classes = ['car', 'cargroup', 'person', 'persongroup', 'traffic sign']
    cropping_ios_threshold = 0.5
    matching_iou_threshold = 0.5
    random_brighten_ratio = 0.8
    num_prior_bbox, imgWidth, imgHeight, crop_coordinate = None, None, None, None

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

        # Initialize variables.
        ratio = (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)
        self.num_prior_bbox = len(ratio) + 1
        self.prior_layer_cfg = [
            {'layer_name': 'Layer7', 'feature_dim_hw': (75, 75), 'bbox_size': (7.5, 7.5), 'aspect_ratio': ratio},
            {'layer_name': 'Layer11', 'feature_dim_hw': (38, 38), 'bbox_size': (40.71, 40.71), 'aspect_ratio': ratio},
            {'layer_name': 'Layer23', 'feature_dim_hw': (19, 19), 'bbox_size': (73.93, 73.93), 'aspect_ratio': ratio},
            {'layer_name': 'Layer27', 'feature_dim_hw': (10, 10), 'bbox_size': (107.14, 107.14), 'aspect_ratio': ratio},
            {'layer_name': 'Layer29', 'feature_dim_hw': (5, 5), 'bbox_size': (140.36, 140.36), 'aspect_ratio': ratio},
            {'layer_name': 'Layer31', 'feature_dim_hw': (3, 3), 'bbox_size': (173.57, 173.57), 'aspect_ratio': ratio},
            {'layer_name': 'Layer33', 'feature_dim_hw': (2, 2), 'bbox_size': (206.79, 206.79), 'aspect_ratio': ratio},
            {'layer_name': 'Layer35', 'feature_dim_hw': (1, 1), 'bbox_size': (240, 240), 'aspect_ratio': ratio}
        ]

        self.prior_bboxes = bbox_helper.generate_prior_bboxes(prior_layer_cfg=self.prior_layer_cfg)

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
        image = self.brighten(image)
        image = self.normalize(image)
        image = np.array(image)
        image = torch.Tensor(image)

        # Prepare labels second to apply crop.
        self.raw_labels = self.crop(self.raw_labels)
        labels = self.normalize(self.raw_labels)
        labels = np.array(labels)

        # Do the matching prior and generate ground-truth labels as well as the boxes.
        self.oracle_labels = labels
        labels = bbox_helper.match_priors(self.prior_bboxes, labels, iou_threshold=self.matching_iou_threshold)

        confidences = labels[:, :len(self.classes)]
        confidences = np.array(confidences)
        confidences = torch.Tensor(confidences)
        locations = labels[:, len(self.classes):]
        locations = np.array(locations)
        locations = torch.Tensor(locations)

        return image, confidences, locations

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

            # Check the ios of the cropped image with oracle bounding box to ensure at least one labeled item.
            found = False
            while not found:
                crop = random.randint(0, self.imgWidth - 301)
                self.crop_coordinate = [crop, 0, crop + 300, 300]
                for label in self.raw_labels:
                    a = torch.Tensor([label[-4:]])
                    b = torch.Tensor([bbox_helper.corner2center(self.crop_coordinate)])

                    if bbox_helper.ios(a, b) > self.cropping_ios_threshold:
                        found = True
                        image = image.crop(self.crop_coordinate)
                        break

            return np.asarray(image)

        # Case for label input.
        labels = inp
        labels[:, len(self.classes)] -= self.crop_coordinate[0]

        # Remove label with too small ios.
        ios = bbox_helper.ios(torch.Tensor(labels[:, len(self.classes):]), torch.Tensor([[150, 150, 300, 300]]))
        labels[np.where(ios <= self.cropping_ios_threshold)] = 0
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

    def brighten(self, image):
        sign = [-1, 1][random.randrange(2)]
        img = np.multiply(image, (1 + sign * (random.uniform(0, self.random_brighten_ratio))))
        return img.clip(0, 255)

    def normalize(self, inp):
        # Case for image input.
        if inp.shape == (300, 300, 3):
            image = inp
            image = np.subtract(image, self.mean)
            image = np.true_divide(image, self.std)

            return image

        # Case for label input.
        labels = inp
        classes = labels[:, 0:len(self.classes)]
        bboxes = labels[:, len(self.classes):]
        labels = np.concatenate((classes, np.true_divide(bboxes, 300.)), axis=1)

        return labels

    def denormalize(self, inp):
        # Denormalize the image.
        inp = np.array(inp, dtype=float)
        if inp.shape == (300, 300, 3):
            image = inp
            image = np.multiply(image, self.std)
            image = np.add(image, self.mean)

            return image.astype(int)

        # Denormalize the landmarks.
        labels = inp
        classes = labels[:, :len(self.classes)]
        bboxes = labels[:, len(self.classes):]
        labels = np.concatenate((classes, np.multiply(bboxes, 300.)), axis=1)

        return labels

    def preview(self, index=-1):
        if index == -1:
            index = random.randint(0, len(self.dataset_list))

        # Acquire preview targets.
        image, confidences, locations = self[index]
        image = np.array(image)
        labels = np.concatenate((np.array(confidences), np.array(locations)), axis=1)

        # Denormalize the data.
        image = self.denormalize(image)
        labels = self.denormalize(labels)

        # Remove labels with no matched class.
        for i_label in range(labels.shape[0]):
            if np.max(labels[i_label, 0:-4]) == 0:
                labels[i_label] = 0
        labels = labels[~np.all(labels == 0, axis=1)]

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        matched_rect, oracle_rect = None, None
        # Display matched bounding boxes.
        for i_label in range(labels.shape[0]):
            corner = bbox_helper.center2corner(labels[i_label, len(self.classes):])
            x = corner[0]
            y = corner[1]
            matched_rect = patches.Rectangle(
                (x, y),
                labels[i_label][len(self.classes) + 2],
                labels[i_label][len(self.classes) + 3],
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(matched_rect)

        # Display ground truth bounding boxes.
        for i_label in range(self.raw_labels.shape[0]):
            corner = bbox_helper.center2corner(self.raw_labels[i_label, len(self.classes):])
            x = corner[0]
            y = corner[1]
            oracle_rect = patches.Rectangle(
                (x, y),
                self.raw_labels[i_label][len(self.classes) + 2],
                self.raw_labels[i_label][len(self.classes) + 3],
                linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(oracle_rect)

        fig.canvas.set_window_title('Preview at Index [' + str(index) + ']')
        plt.title('Preview at Index [' + str(index) + ']')
        plt.xlim(0, 300)
        plt.ylim(300, 0)
        plt.legend([oracle_rect, matched_rect],
                   ['Oracle Box', 'Matched Box'],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()

        # Output stats.
        print('Available classes:', self.classes)
        print('Number of oracle bounding boxes:', self.raw_labels.shape[0])
        print('Number of matched bounding boxes:', labels.shape[0])
        print('Bounding box configuration:')
        print(np.array(self.prior_layer_cfg))

        plt.show()
