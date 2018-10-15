import torch.nn as nn
import torch.nn.functional as f
import torch


def hard_negative_mining(predicted_prob, oracle_label, neg_pos_ratio=3.0):
    """
    The training sample has much more negative samples, the hard negative mining and produce balanced 
    positive and negative examples.
    :param predicted_prob: predicted probability for each prior item, dim: (N, H * W * num_prior)
    :param oracle_label: ground_truth label, dim: (N, H * W * num_prior)
    :param neg_pos_ratio:
    :return:
    """
    pos_flag = oracle_label > 0                                    # 0 = negative label

    # Sort the negative samples.
    predicted_prob[pos_flag] = -1.0                                # Temporarily remove positive by setting -1
    _, indices = predicted_prob.sort(dim=1, descending=True)       # Sort by descend order, the positives are at the end
    _, orders = indices.sort(dim=1)                                # Sort the negative samples by its original index

    # Remove the extra negative samples.
    num_pos = pos_flag.sum(dim=1, keepdim=True)                    # Compute the num. of positive examples
    num_neg = neg_pos_ratio * num_pos                              # Determine of neg. examples, should < neg_pos_ratio
    neg_flag = orders < num_neg                                    # Retain the first 'num_neg' negative samples index.

    return pos_flag, neg_flag


class MultiboxLoss(nn.Module):

    def __init__(self, bbox_pre_var, iou_threshold=0.5, neg_pos_ratio=3.0):
        super(MultiboxLoss, self).__init__()
        self.bbox_center_var, self.bbox_size_var = bbox_pre_var[:2], bbox_pre_var[2:]
        self.iou_thres = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_label_idx = 0

    def forward(self, confidence, pred_loc, oracle_class_labels, oracle_bbox_loc):
        """
         Compute the Multibox joint loss:
            L = (1/N) * L_{loc} + L_{class}
        :param confidence: predicted class probability, dim: (N, H*W*num_prior, num_classes)
        :param pred_loc: predicted prior bounding boxes, dim: (N, H*W*prior_num, 4)
        :param oracle_class_labels: ground-truth class label, dim:(N, H*W*num_prior)
        :param oracle_bbox_loc: ground-truth bounding box for prior, dim: (N, H*W*num_prior, 4)
        :return:
        """
        # Do the hard negative mining and produce balanced positive and negative examples.
        with torch.no_grad():
            neg_class_prob = -f.log_softmax(confidence, dim=2)[:, :, self.neg_label_idx]  # Select neg. class prob.
            pos_flag, neg_flag = \
                hard_negative_mining(neg_class_prob, oracle_class_labels, neg_pos_ratio=self.neg_pos_ratio)
            sel_flag = pos_flag | neg_flag
            num_pos = pos_flag.sum(dim=1, keepdim=True)

        # Loss for the classification.
        num_classes = confidence.shape[2]
        sel_conf = confidence[sel_flag]
        conf_loss = f.cross_entropy(sel_conf.reshape(-1, num_classes), oracle_class_labels[sel_flag]) / num_pos

        # Loss for the bounding box prediction.
        loc_huber_loss = None
        # TODO: implementation on bounding box regression

        return conf_loss, loc_huber_loss