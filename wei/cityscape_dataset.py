import random
import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset
import PIL
import PIL.PngImagePlugin
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import bbox_helper as helper
import cProfile


# noinspection PyUnresolvedReferences
class CityScapeDataset(Dataset):
    # Configurations.
    classes = ['car', 'cargroup', 'person', 'persongroup']
    bounding_box_ratios = (1.0, 1/2, 1/3, 1/4, 2.0, 3.0, 4.0)
    matching_iou_threshold = 0.3
    cropping_ios_threshold = 0.5
    random_brighten_ratio = 0.5
    s_min = 0.025
    s_max = 0.8
    is_debug = False

    def __init__(self, dataset_list, num_worker):
        self.dataset_list = dataset_list
        self.num_prior_bbox = len(self.bounding_box_ratios) + 1
        self.prepared_index = 0
        self.num_worker = num_worker

        # Initialize variables.
        self.imgWidth, self.imgHeight, self.crop_coordinates, self.locations = None, None, None, None
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
        self.prior_bboxes = helper.generate_prior_bboxes(prior_layer_cfg=self.prior_layer_cfg)  # 60552.

        # Pre-process parameters, normalize: (I-self.mean)/self.std.
        self.mean = torch.Tensor([127, 127, 127])
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
        self.prepared_index += self.num_worker
        current_index = (self.prepared_index % len(self.dataset_list))
        n_instance = ('[{:' + digit + 'd}').format(current_index) + '/' + str(len(self.dataset_list)) + ']'
        n_percentage = '[{:6.2f}%]'.format(current_index * 100. / len(self.dataset_list))
        print('\r' + ('Preparing dataset at index [{:' + digit + 'd}').format(index) + ']'
              + n_instance + n_percentage, end='')

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
        image = Image.open(item['file'])
        confidences, locations = self.sanitize(item)

        print(np.array(image).shape)
        print('PRINT1')

        # Return the case there is no match at all.
        if confidences.nonzero().shape[0] == 0:
            image = self.resize(image)
            image = self.crop(image)
            image = self.normalize(image)
            image = image.view((image.shape[2], image.shape[0], image.shape[1]))

            print('OUT SHAPE')
            print(image.shape, confidences.shape)

            return image, confidences, locations

        print('PRINT2')

        # Resize the image and label first.
        image = self.resize(image)
        locations = self.resize(locations)

        print('PRINT3')

        # Prepare image array first to update crop.
        image = self.crop(image)
        image = self.brighten(image)
        image = self.normalize(image)

        print('PRINT4')

        # Prepare labels second to apply crop.
        locations = self.crop(locations)
        locations = self.normalize(locations)

        print('PRINT5')

        # Do the matching prior and generate ground-truth labels as well as the boxes.
        confidences = helper.match_priors(
            self.prior_bboxes, locations, iou_threshold=self.matching_iou_threshold)

        if self.is_debug:
            pr.disable()
            pr.print_stats(sort='time')

        # Reshape image to channel by X by Y.
        image = image.view((image.shape[2], image.shape[0], image.shape[1]))

        print('PRINT6')
        print('OUT SHAPE')
        print(image.shape, confidences.shape)

        return image, confidences, self.prior_bboxes

    def sanitize(self, item):
        confidences = []
        locations = []

        for obj in item['objects']:
            confidence = torch.zeros(len(self.classes))

            try:
                confidence[self.classes.index(obj['label'])] = 1.0
                polygons = torch.Tensor(obj['polygon'])
                corner = [min(polygons[:, 0]), min(polygons[:, 1]), max(polygons[:, 0]), max(polygons[:, 1])]
                corner = torch.stack(corner)

                # Add confidences and locations.
                confidences.append(confidence)
                locations.append(helper.corner2center(corner))
            except ValueError:
                pass

        # Detect if there is nothing found.
        if len(confidences) == 0:
            confidences = torch.zeros([len(self.prior_bboxes)])
            locations = self.prior_bboxes
        else:
            confidences = torch.stack(confidences)
            locations = torch.stack(locations)

        return confidences, locations

    def resize(self, inp):
        # Case for image input.
        if isinstance(inp, PIL.PngImagePlugin.PngImageFile):
            image = inp
            if self.imgWidth < self.imgHeight:
                self.imgWidth = 300
                self.imgHeight = int(self.imgHeight / self.resize_ratio)
            else:
                self.imgWidth = int(self.imgWidth / self.resize_ratio)
                self.imgHeight = 300
            image = image.resize((self.imgWidth, self.imgHeight), Image.ANTIALIAS)
            image = np.array(image)
            return torch.Tensor(image)

        # Case for location input.
        locations = inp
        locations = torch.div(locations, self.resize_ratio)
        self.locations = locations
        return locations

    def crop(self, inp):
        # Case for image input.
        if inp.shape == torch.Size([self.imgHeight, self.imgWidth, 3]):
            image = inp

            # Return 300x300 patch if no object is detected.
            if self.locations is None:
                return image[0:300, 0:300, :]

            # Check the ios of the cropped image with oracle bounding box to ensure at least one labeled item.
            found = False
            while not found:
                crop = random.randint(0, self.imgWidth - 300)
                self.crop_coordinates = torch.Tensor([crop, 0, crop + 300, 300])
                for location in self.locations:
                    if helper.ios(location, helper.corner2center(self.crop_coordinates)) > self.cropping_ios_threshold:
                        found = True
                        image = image[
                                int(self.crop_coordinates[1]):int(self.crop_coordinates[3]),
                                int(self.crop_coordinates[0]):int(self.crop_coordinates[2]), :]
                        break

            return image

        # Case for location input.
        locations = inp
        locations[:, 0] -= self.crop_coordinates[0]

        # Set locations to 0 if the ios is too small.
        ios = helper.ios(locations, torch.Tensor([150, 150, 300, 300]))
        locations[ios <= self.cropping_ios_threshold] = 0

        # Clip the location.
        locations = helper.center2corner(locations)
        locations = torch.clamp(locations, 0, 300)
        locations = helper.corner2center(locations)

        # Save the oracle locations.
        self.locations = locations

        return locations

    def brighten(self, image):
        sign = [-1, 1][random.randrange(2)]
        image = torch.mul(image, (1 + sign * (random.uniform(0, self.random_brighten_ratio))))
        return torch.clamp(image, 0, 255)

    def normalize(self, inp):
        # Case for image input.
        if inp.shape == torch.Size([300, 300, 3]):
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
        if inp.shape == torch.Size([300, 300, 3]):
            image = inp
            image = torch.mul(image, self.std)
            image = torch.add(image, self.mean)

            return image

        # Denormalize the location.
        locations = inp
        locations = torch.mul(locations, 300.)

        return locations

    def preview(self, index=0):
        if index == -1:
            index = random.randint(0, len(self.dataset_list))

        # Acquire preview targets.
        image, confidences, locations = self[index]
        image = image.view(image.shape[1], image.shape[2], image.shape[0])

        # Denormalize the data.
        image = self.denormalize(image)
        locations = self.denormalize(locations)

        # Prepare data for plotting.
        image = np.array(image).astype(int)
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        matched_rect, oracle_rect = None, None
        # Display matched bounding boxes.
        for i_location in range(0, locations.shape[0]):
            if confidences[i_location] != 0:
                corner = helper.center2corner(locations[i_location])
                x = corner[0]
                y = corner[1]
                matched_rect = patches.Rectangle(
                    (x, y),
                    locations[i_location][2],
                    locations[i_location][3],
                    linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(matched_rect)

        # Display ground truth bounding boxes.
        for i_location in range(0, self.locations.shape[0]):
            corner = helper.center2corner(self.locations[i_location])
            x = corner[0]
            y = corner[1]
            oracle_rect = patches.Rectangle(
                (x, y),
                self.locations[i_location][2],
                self.locations[i_location][3],
                linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(oracle_rect)

        fig.canvas.set_window_title('Preview at Index [' + str(index) + ']')
        plt.title('Preview at Index [' + str(index) + ']')
        plt.xlim(0, 300)
        plt.ylim(300, 0)
        if matched_rect is not None:
            plt.legend([oracle_rect, matched_rect],
                       ['Oracle Box', 'Matched Box'],
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            plt.legend([oracle_rect],
                       ['Oracle Box'],
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()

        # Output stats.
        print('Available classes:', self.classes)
        print('Number of oracle bounding boxes:', self.locations.shape[0])
        print('Number of matched bounding boxes:', locations.shape[0])
        print('Bounding box configuration:')
        print(np.array(self.prior_layer_cfg))

        plt.show()
