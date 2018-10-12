import torch
import torch.nn as nn
import torch.nn.functional as f
import module_util
from mobilenet import MobileNet


class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # Setup the backbone network (base_net).
        self.base_net = MobileNet()

        # The feature map will extracted from the end of following layers sections in (base_net).
        self.base_output_sequence_indices = (0, 6, 12, len(self.base_net.base_net))

        # Number of prior bounding box.
        self.num_prior_bbox = 6

        # Define the additional feature extractor.
        self.additional_feature_extractor = nn.ModuleList([
            # Layer 28 - 29 5x5x512
            nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            # Layer 30 - 31 3x3x256
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            # Layer 32 - 33 2x2x256
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            # Layer 34 - 35 1x1x256
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        ])

        # Bounding box offset regressor.
        self.loc_regressor = nn.ModuleList([
            nn.Conv2d(256, self.num_prior_bbox * 4, kernel_size=3, padding=1),               # Layer 11
            nn.Conv2d(512, self.num_prior_bbox * 4, kernel_size=3, padding=1),               # Layer 22
            nn.Conv2d(1024, self.num_prior_bbox * 4, kernel_size=3, padding=1),              # Layer 27
            nn.Conv2d(512, self.num_prior_bbox * 4, kernel_size=3, padding=1),               # Layer 29
            nn.Conv2d(256, self.num_prior_bbox * 4, kernel_size=3, padding=1),               # Layer 31
            nn.Conv2d(256, self.num_prior_bbox * 4, kernel_size=3, padding=1),               # Layer 33
            nn.Conv2d(256, self.num_prior_bbox * 4, kernel_size=3, padding=1),               # Layer 35
        ])

        # Bounding box classification confidence for each label.
        self.classifier = nn.ModuleList([
            nn.Conv2d(256, self.num_prior_bbox * num_classes, kernel_size=3, padding=1),     # Layer 11
            nn.Conv2d(512, self.num_prior_bbox * num_classes, kernel_size=3, padding=1),     # Layer 13
            nn.Conv2d(1024, self.num_prior_bbox * num_classes, kernel_size=3, padding=1),    # Layer 25
            nn.Conv2d(512, self.num_prior_bbox * num_classes, kernel_size=3, padding=1),     # Layer 29
            nn.Conv2d(256, self.num_prior_bbox * num_classes, kernel_size=3, padding=1),     # Layer 31
            nn.Conv2d(256, self.num_prior_bbox * num_classes, kernel_size=3, padding=1),     # Layer 33
            nn.Conv2d(256, self.num_prior_bbox * num_classes, kernel_size=3, padding=1),     # Layer 35
        ])

        # Load pretrained model.
        pretrained_state = torch.load('pretrained/mobienetv2.pth')
        model_dict = self.base_net.state_dict()

        # Filter out unnecessary keys.
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}

        # Overwrite entries in the existing state dict.
        model_dict.update(pretrained_state)

        # Load the new state dict.
        self.base_net.load_state_dict(model_dict)

        def init_with_xavier(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        self.loc_regressor.apply(init_with_xavier)
        self.classifier.apply(init_with_xavier)
        self.additional_feature_extractor.apply(init_with_xavier)

    def feature_to_bbox(self, loc_regress_layer, confidence_layer, input_feature):
        """
        Compute the bounding box class scores and the bounding box offset.
        :param loc_regress_layer: offset regressor layer to run forward.
        :param confidence_layer: confidence layer to run forward.
        :param input_feature: feature map to be feed in forward.
        :return: confidence and location, with dim:(N, num_priors, num_classes) and dim:(N, num_priors, 4) respectively.
        """
        conf = confidence_layer(input_feature)
        loc = loc_regress_layer(input_feature)

        # Confidence post-processing:
        # 1: (N, num_prior_bbox * n_classes, H, W) to
        # (N, H * W * num_prior_bbox, n_classes) = (N, num_priors, num_classes)
        # where H * W * num_prior_bbox = num_priors
        conf = conf.permute(0, 2, 3, 1).contiguous()
        num_batch = conf.shape[0]
        c_channels = int(conf.shape[1] * conf.shape[2] * conf.shape[3] / self.num_classes)
        conf = conf.view(num_batch, c_channels, self.num_classes)

        # Bounding Box loc and size post-processing.
        # 1: (N, num_prior_bbox * 4, H, W) to (N, num_priors, 4)
        loc = loc.permute(0, 2, 3, 1).contiguous()
        loc = loc.view(num_batch, c_channels, 4)

        return conf, loc

    def forward(self, inp):

        confidence_list = []
        loc_list = []
        result = inp

        # Forward the 'result' to base net for regressor & classifier.
        for index in range(0, len(self.base_output_sequence_indices) - 1):
            result = module_util.forward_from(
                self.base_net.base_net,
                self.base_output_sequence_indices[index], self.base_output_sequence_indices[index + 1], result)
            confidence, loc = self.feature_to_bbox(self.loc_regressor[index], self.classifier[index], result)
            confidence_list.append(confidence)
            loc_list.append(loc)

        # Forward the 'result' to additional layers for extracting coarse features.
        for index in range(0, len(self.additional_feature_extractor)):
            result = module_util.forward_from(
                self.additional_feature_extractor,
                index, index + 1, result)
            confidence, loc = self.feature_to_bbox(self.loc_regressor[index + 3], self.classifier[index + 3], result)
            confidence_list.append(confidence)
            loc_list.append(loc)

        confidences = torch.cat(confidence_list, 1)
        locations = torch.cat(loc_list, 1)

        # [Debug] Check the output.
        print(confidences.shape)
        assert confidences.dim() == 3                       # Should be (N, num_priors, num_classes).
        assert confidences.shape[2] == self.num_classes     # Should be (N, num_priors, num_classes).
        assert locations.dim() == 3                         # Should be (N, num_priors, 4).
        assert confidences.shape[1] == locations.shape[1]
        assert locations.shape[2] == 4

        if not self.training:
            # If in testing/evaluating mode, normalize the output with Softmax.
            confidences = f.softmax(confidences, dim=1)

        return confidences, locations