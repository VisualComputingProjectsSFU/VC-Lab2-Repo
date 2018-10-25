import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np


def hard_negative_mining(confidence_predictions, confidence_oracles, neg_pos_ratio=3.0):
    """
    The training sample has much more negative samples, the hard negative mining and produce balanced
    positive and negative examples.
    :param confidence_predictions: predicted probability for each prior item, dim: (N, H * W * num_prior)
    :param confidence_oracles: ground_truth label, dim: (N, H * W * num_prior)
    :param neg_pos_ratio:
    :return:
    """
    pos_flags = confidence_oracles > 0                              # 0 = negative label.

    # Sort the negative samples.
    confidence_predictions[pos_flags] = 0                           # Temporarily remove positive by setting 0.
    confidence_predictions = torch.abs(confidence_predictions)
    confidence_predictions = torch.sum(confidence_predictions, -1)

    _, indices = confidence_predictions.sort(descending=True)       # Sort descend order.
    _, orders = indices.sort()                                      # Sort the negative samples by its original index.

    # Remove the extra negative samples.
    num_pos = int(pos_flags.sum())                                  # Compute the num. of positive examples.
    num_neg = int(neg_pos_ratio * num_pos)                          # Determine neg. examples, should < neg_pos_ratio.
    neg_flags = orders < num_neg                                    # Retain the first 'num_neg' negative samples index.

    return pos_flags, neg_flags


class MultiboxLoss(nn.Module):

    def __init__(self, iou_threshold=0.5, neg_pos_ratio=3.0):
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence_predictions, location_predictions, confidence_oracles, location_oracles):
        """
         Compute the Multibox joint loss:
            L = (1/N) * L_{loc} + L_{class}
        :param confidence_predictions: predicted class probability, dim: (N, H*W*num_prior, num_classes)
        :param location_predictions: predicted prior bounding boxes, dim: (N, H*W*prior_num, 4)
        :param confidence_oracles: ground-truth class label, dim:(N, H*W*num_prior)
        :param location_oracles: ground-truth bounding box for prior, dim: (N, H*W*num_prior, 4)
        :return:
        """
        # Return -1 if no positive match is found.
        if confidence_oracles.nonzero().shape[0] == 0:
            return -1

        # Do the hard negative mining and produce balanced positive and negative examples.
        with torch.no_grad():
            confidence_predictions_normalized = f.softmax(confidence_predictions, dim=-1)
            pos_flags, neg_flags = hard_negative_mining(
                confidence_predictions_normalized,
                confidence_oracles,
                neg_pos_ratio=self.neg_pos_ratio)
            selected_flags = pos_flags.type(torch.ByteTensor) | neg_flags.type(torch.ByteTensor)
            num_positive = pos_flags.data.sum().float()

        if num_positive == 0.0:
            return torch.Tensor[0]

        # Loss for the classification.
        confidence_loss = f.cross_entropy(
            confidence_predictions[selected_flags].requires_grad_(),
            confidence_oracles[selected_flags].type(torch.cuda.LongTensor), reduction='sum')


        # Loss for the bounding box prediction.
        location_loss = f.smooth_l1_loss(
            location_predictions[pos_flags],
            location_oracles[pos_flags].type(torch.cuda.FloatTensor), reduction='sum')
        loss = torch.div(torch.add(confidence_loss, location_loss), num_positive)

        return loss