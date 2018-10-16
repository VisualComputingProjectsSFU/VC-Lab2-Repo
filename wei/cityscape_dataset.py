import os
import random
import numpy as np
import torch.nn
import torch.cuda
from torch.utils.data import Dataset
import PIL
import PIL.PngImagePlugin
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import train
import bbox_helper as helper
import cProfile


# noinspection PyUnresolvedReferences
class CityScapeDataset(Dataset):

    # Configurations.
    classes = ['car', 'cargroup', 'person', 'persongroup']
    bounding_box_ratios = (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)
    matching_iou_threshold = 0.5
    cropping_ios_threshold = 0.5
    random_brighten_ratio = 0.5
    s_min = 0.025
    s_max = 0.8
    is_debug = True

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.prepared_index = 1
        self.num_prior_bbox = len(self.bounding_box_ratios) + 1

        # Configure default torch tensor settings.
        if torch.cuda.is_available():
            print("GPU Acceleration Enabled")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.cuda.set_device(0)
            torch.multiprocessing.set_start_method('spawn')

        # Initialize variables.
        self.imgWidth, self.imgHeight, self.crop_coordinate = None, None, None
        self.prior_layer_cfg = [
            {'feature_dim_hw': (75, 75), 'bbox_size': (7.5, 7.5), 'aspect_ratio': self.bounding_box_ratios},
            {'feature_dim_hw': (38, 38), 'bbox_size': (40.71, 40.71), 'aspect_ratio': self.bounding_box_ratios},
            {'feature_dim_hw': (19, 19), 'bbox_size': (73.93, 73.93), 'aspect_ratio': self.bounding_box_ratios},
            {'feature_dim_hw': (10, 10), 'bbox_size': (107.14, 107.14), 'aspect_ratio': self.bounding_box_ratios},
            {'feature_dim_hw': (5, 5), 'bbox_size': (140.36, 140.36), 'aspect_ratio': self.bounding_box_ratios},
            {'feature_dim_hw': (3, 3), 'bbox_size': (173.57, 173.57), 'aspect_ratio': self.bounding_box_ratios},
            {'feature_dim_hw': (2, 2), 'bbox_size': (206.79, 206.79), 'aspect_ratio': self.bounding_box_ratios},
            {'feature_dim_hw': (1, 1), 'bbox_size': (240, 240), 'aspect_ratio': self.bounding_box_ratios}
        ]
        self.prior_bboxes = helper.generate_prior_bboxes(prior_layer_cfg=self.prior_layer_cfg)

        # Pre-process parameters, normalize: (I-self.mean)/self.std.
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes. Labels are include
        [car, traffic sign, person]. irrelevant objects are set to 0.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4).
        :return bbox_label: matched classification label, dim: (num_priors).
        """
        # Alert current dataset status.
        digit = str(len(str(len(self.dataset_list))))
        n_instance = ('[{:' + digit + 'd}').format(self.prepared_index) + '/' + str(len(self.dataset_list)) + ']'
        n_percentage = '[{:6.2f}%]'.format(self.prepared_index * 100. / len(self.dataset_list))
        print(('Preparing dataset at index [{:' + digit + 'd}').format(index) + ']' + n_instance + n_percentage)

        if self.is_debug:
            pr = cProfile.Profile()
            pr.enable()
        else:
            pr = None

        # Prepare configurations.
        item = self.dataset_list[index]
        self.imgWidth = float(item['imgWidth'])
        self.imgHeight = float(item['imgHeight'])
        self.resize_ratio = min(self.imgHeight / 300., self.imgWidth / 300.)
        file_path = os.path.join(train.data_path, item['file'])
        image = Image.open(file_path)
        confidences, locations = self.sanitize(item)

        # Resize the image and label first.
        image = self.resize(image)
        locations = self.resize(locations)

        # Prepare image array first to update crop.
        image = self.crop(image)
        image = self.brighten(image)
        image = self.normalize(image)

        # Prepare labels second to apply crop.
        locations = self.crop(locations)
        locations = self.normalize(locations)

        # Do the matching prior and generate ground-truth labels as well as the boxes.
        self.oracle_locations = locations
        confidences, locations = helper.match_priors(
            self.prior_bboxes, confidences, locations, iou_threshold=self.matching_iou_threshold)

        self.prepared_index += 1

        if self.is_debug:
            pr.disable()
            pr.print_stats(sort='time')

        return image, confidences, locations

    def sanitize(self, item):
        confidences = []
        locations = []

        for obj in item['objects']:
            confidence = torch.zeros(len(self.classes)).cuda()

            try:
                confidence[self.classes.index(obj['label'])] = 1.0
                polygons = torch.Tensor(obj['polygon'])
                corner = [min(polygons[:, 0]), min(polygons[:, 1]), max(polygons[:, 0]), max(polygons[:, 1])]

                # Add confidences and locations.
                confidences.append(confidence)
                locations.append(helper.corner2center(corner))
            except ValueError:
                pass

        return confidences, locations

    def resize(self, inp):
        # Case for image input.
        if isinstance(inp, PIL.PngImagePlugin.PngImageFile):
            image = inp
            if self.imgWidth < self.imgHeight:
                self.imgWidth = 300
                self.imgHeight = self.imgHeight / self.resize_ratio
            else:
                self.imgWidth = self.imgWidth / self.resize_ratio
                self.imgHeight = 300
            image = image.resize((int(self.imgWidth), int(self.imgHeight)), Image.ANTIALIAS)
            image = np.array(image)
            return torch.Tensor(image)

        # Case for location input.
        locations = torch.Tensor(inp)
        locations = torch.div(locations, self.resize_ratio)
        return locations

    def crop(self, inp):
        # Case for image input.
        if inp.shape == (300, 300, 3):
            image = inp

            # Check the ios of the cropped image with oracle bounding box to ensure at least one labeled item.
            found = False
            while not found:
                crop = random.randint(0, self.imgWidth - 300)
                self.crop_coordinate = torch.Tensor((crop, 0, crop + 300, 300))
                for location in self.oracle_locations:
                    if helper.ios(location, helper.corner2center(self.crop_coordinate)) > self.cropping_ios_threshold:
                        found = True
                        image = image[
                                self.crop_coordinate[0]:self.crop_coordinate[1],
                                self.crop_coordinate[2]:self.crop_coordinate[3], :]
                        break

            return image

        # Case for location input.
        locations = inp
        locations[:, len(self.classes)] -= self.crop_coordinate[0]

        # Remove label with too small ios.
        ios = helper.ios(locations, torch.Tensor([[150, 150, 300, 300]]))
        locations[np.where(ios <= self.cropping_ios_threshold)] = 0
        # TODO: Remove rows with all zeros.

        # Clip the location.
        # TODO: Clamp.
        for i_location in range(0, locations.shape[0]):
            corner = helper.center2corner(locations[i_location])
            corner[0] = max(corner[0], 0)
            corner[1] = min(corner[1], 300)
            corner[2] = max(corner[2], 0)
            corner[3] = min(corner[3], 300)
            center = helper.corner2center(corner)
            locations[i_location] = center

        return locations

    def brighten(self, image):
        sign = [-1, 1][random.randrange(2)]
        image = torch.mul(image, (1 + sign * (random.uniform(0, self.random_brighten_ratio))))
        return torch.clamp(image, 0, 255)

    def normalize(self, inp):
        # Case for image input.
        if inp.shape == (300, 300, 3):
            image = inp
            image = torch.sub(image, self.mean)
            image = torch.div(image, self.std)

            return image

        # Case for location input.
        locations = inp
        locations = torch.div(locations, 300.)

        return locations

    def denormalize(self, inp):
        # Denormalize the image.
        if inp.shape == (300, 300, 3):
            image = inp
            image = torch.mul(image, self.std)
            image = torch.add(image, self.mean)

            return image

        # Denormalize the location.
        locations = inp
        locations = torch.mul(locations, 300.)

        return locations

    def preview(self, index=-1):
        if index == -1:
            index = random.randint(0, len(self.dataset_list))

        # Acquire preview targets.
        image, confidences, locations = self[index]
        image = np.array(image)

        # Denormalize the data.
        image = self.denormalize(image)
        locations = self.denormalize(locations)

        # TODO:
        # Remove labels with no matched class.
        for i_label in range(locations.shape[0]):
            if np.max(locations[i_label, 0:-4]) == 0:
                locations[i_label] = 0
        labels = locations[~np.all(locations == 0, axis=1)]

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        matched_rect, oracle_rect = None, None
        # Display matched bounding boxes.
        for i_label in range(labels.shape[0]):
            corner = helper.center2corner(labels[i_label, len(self.classes):])
            x = corner[0]
            y = corner[1]
            matched_rect = patches.Rectangle(
                (x, y),
                labels[i_label][len(self.classes) + 2],
                labels[i_label][len(self.classes) + 3],
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(matched_rect)

        # Display ground truth bounding boxes.
        for i_label in range(self.oracle_confidences.shape[0]):
            corner = helper.center2corner(self.oracle_confidences[i_label, len(self.classes):])
            x = corner[0]
            y = corner[1]
            oracle_rect = patches.Rectangle(
                (x, y),
                self.oracle_confidences[i_label][len(self.classes) + 2],
                self.oracle_confidences[i_label][len(self.classes) + 3],
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
        print('Number of oracle bounding boxes:', self.oracle_confidences.shape[0])
        print('Number of matched bounding boxes:', labels.shape[0])
        print('Bounding box configuration:')
        print(np.array(self.prior_layer_cfg))

        plt.show()
