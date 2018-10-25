import numpy as np
import torch.nn
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors
from PIL import Image
import bbox_helper as helper
from random import randint
import matplotlib.pyplot as plt
import random
import PIL.PngImagePlugin


class CityScapeDataset(Dataset):
    cropping_ios_threshold = 0.5
    random_brighten_ratio = 0.5

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.image_size = 300
        self.imgWidth, self.imgHeight, self.crop_coordinate = None, None, None
        self.ios_index =None

        # TODO: implement prior bounding box
        """
            Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

            Use VGG_SSD 300x300 as example:
            Feature map dimension for each output layers:
               Layer     | Map Dim (h, w)  | Single bbox size that covers in the original image
            1. Conv6     | (38x38)         | (30x30) (unit. pixels)
            1. Conv11    | (19x19)        | (60x60) (unit. pixels)
            2. Conv13    | (10x10)        | (113x113)
            3. Conv14_2  | (5x5)          | (165x165)
            4. Conv15_2  | (3x3)          | (218x218)
            5. Conv16_2  | (1x1)          | (270x270)
            6. Conv17_2  | (1x1)          | (264x264)
            NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
            Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
            :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
            :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
            """
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
        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg)
        # Pre-process parameters:
        #  Normalize: (I-self.mean)/self.std
        self.mean = torch.Tensor([127, 127, 127])
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """

        # TODO: implement data loading
        # 1. Load image as well as the bounding box with its label
        # 2. Normalize the image with self.mean and self.std
        # 3. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        # 4. Normalize the bounding box position value from 0 to 1
        item = self.dataset_list[idx]
        image_path = item['image_path']
        labels = np.asarray(item['labels'])
        labels = torch.Tensor(labels).cuda()
        locations = torch.Tensor(item['bboxes']).cuda()
        bbox = np.array(item['bboxes'])

        image = Image.open(image_path)

        self.imgWidth, self.imgHeight = image.size
        self.resize_ratio = min(self.imgHeight / 300., self.imgWidth / 300.)

        locations = helper.corner2center(locations)


        image = self.resize(image)
        locations = self.resize(locations)


        # Prepare image array first to update crop.
        image = self.crop(image)
        image = self.brighten(image)
        image = self.normalize(image)


        # Prepare labels second to apply crop.
        locations = self.crop(locations)
        locations = self.normalize(locations)

        # convert to tensor
        img_tensor = image.view((image.shape[2], image.shape[0], image.shape[1]))
        img_tensor = img_tensor.cuda()

        labels = labels[self.ios_index]

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box

        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, helper.center2corner(locations), labels,
                                                      iou_threshold=0.5)


        # [DEBUG] check the output.
        # assert isinstance(bbox_label_tensor, torch.Tensor)
        # assert isinstance(bbox_tensor, torch.Tensor)
        # assert bbox_tensor.dim() == 2
        # assert bbox_tensor.shape[1] == 4
        # assert bbox_label_tensor.dim() == 1
        # assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]
        return img_tensor, bbox_tensor, bbox_label_tensor

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
            cnt = 0
            while not found:
                cnt += 1
                if cnt > 300:
                    self.crop_coordinates = torch.Tensor([150, 0, 450, 300])
                    image = image[
                            int(self.crop_coordinates[1]):int(self.crop_coordinates[3]),
                            int(self.crop_coordinates[0]):int(self.crop_coordinates[2]), :]
                    break
                crop = random.randint(0, self.imgWidth - 300)
                self.crop_coordinates = torch.Tensor([crop, 0, crop + 300, 300])
                for location in self.locations:
                    if helper.ios(location,
                                  helper.corner2center(self.crop_coordinates)) > self.cropping_ios_threshold:
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
        self.ios_index = ios > self.cropping_ios_threshold
        locations[ios <= self.cropping_ios_threshold] = 0

        locations = locations[self.ios_index]

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
