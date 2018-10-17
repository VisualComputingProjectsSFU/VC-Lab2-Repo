import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


def hard_negative_mining(predicted_prob, gt_label, neg_pos_ratio=3.0):
    """
    The training sample has much more negative samples, the hard negative mining and produce balanced 
    positive and negative examples.
    :param predicted_prob: predicted probability for each prior item, dim: (N, H*W*num_prior)
    :param gt_label: ground_truth label, dim: (N, H*W*num_prior)
    :param neg_pos_ratio:
    :return:
    """
    pos_flag = gt_label > 0  # 0 = negative label

    # Sort the negative samples
    predicted_prob[pos_flag] = -1.0  # temporarily remove positive by setting -1
    _, indices = predicted_prob.sort(dim=1, descending=True)  # sort by descend order, the positives are at the end
    _, orders = indices.sort(dim=1)  # sort the negative samples by its original index

    # Remove the extra negative samples
    num_pos = pos_flag.sum(dim=1, keepdim=True)  # compute the num. of positive examples
    num_neg = neg_pos_ratio * num_pos  # determine of neg. examples, should < neg_pos_ratio
    neg_flag = orders < num_neg  # retain the first 'num_neg' negative samples index.

    return pos_flag, neg_flag


class MultiboxLoss(nn.Module):
    def __init__(self, bbox_pre_var, iou_threshold=0.5, neg_pos_ratio=3.0):
        super(MultiboxLoss, self).__init__()
        self.bbox_center_var, self.bbox_size_var = bbox_pre_var[:2], bbox_pre_var[2:]
        self.iou_thres = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_label_idx = 0

    def forward(self, confidence, pred_loc, gt_class_labels, gt_bbox_loc):
        """
         Compute the Multibox joint loss:
            L = (1/N) * L_{loc} + L_{class}
        :param confidence: predicted class probability, dim: (N, H*W*num_prior, num_classes)
        :param pred_loc: predicted prior bounding boxes, dim: (N, H*W*prior_num, 4)
        :param gt_class_labels: ground-truth class label, dim:(N, H*W*num_prior)
        :param gt_bbox_loc: ground-truth bounding box for prior, dim: (N, H*W*num_prior, 4)
        :return:
        """
        # Do the hard negative mining and produce balanced positive and negative examples
        # print("in multibox forward")
        # print(confidence.shape)
        # print(pred_loc.shape)
        # print(gt_class_labels.shape)
        # print(gt_bbox_loc.shape)
        batch_size, num_priors, locs = pred_loc.size()
        with torch.no_grad():
            neg_class_prob = -F.log_softmax(confidence, dim=2)[:, :, self.neg_label_idx]  # select neg. class prob.
            pos_flag, neg_flag = hard_negative_mining(neg_class_prob, gt_class_labels, neg_pos_ratio=self.neg_pos_ratio)
            sel_flag = pos_flag | neg_flag
            num_pos = pos_flag.sum(dim=1, keepdim=True).long().float()

        # Loss for the classification
        pos = gt_class_labels > 0  # box matched
        # print(gt_class_labels.shape)
        # print(pred_loc.shape)
        pos_mask = pos.unsqueeze(2).expand_as(confidence)
        print(torch.max(confidence[pos_mask]))

        num_classes = confidence.shape[2]

        mask = []  # avoid divide by 0
        cnt = 0
        for i in range(batch_size):
            if num_pos[i, :] != 0:
                mask.append(i)
                cnt += 1
        if cnt == 0:
            return Variable(torch.Tensor([0])), Variable(torch.Tensor([0]))
        num_pos = num_pos[mask, :]
        sel_flag = sel_flag[mask, :]
        confidence = confidence[mask, :]
        gt_class_labels = gt_class_labels[mask, :]
        sel_conf = confidence[sel_flag]
        err = gt_class_labels[sel_flag] > num_classes
        sum_err = err.sum(dim=0)
        if sum_err > 0:
            print('sum_err',sum_err)
            print(gt_class_labels[sel_flag])
        err1 = gt_class_labels[sel_flag] < 0
        sum_err1 = err1.sum(dim=0)
        if sum_err1 > 0:
            print('sum_err1',sum_err1)
            print(gt_class_labels[sel_flag])
        print(gt_class_labels[sel_flag].shape)
        print(sel_conf.shape)

        conf_loss = F.cross_entropy(sel_conf.reshape(-1, num_classes), gt_class_labels[sel_flag])
        # print('conf_loss',conf_loss)

        # Loss for the bounding box prediction
        # TODO: implementation on bounding box regression

        pos = gt_class_labels > 0  # box matched
        # print(gt_class_labels.shape)
        # print(pred_loc.shape)
        pred_loc = pred_loc[mask, :]
        gt_bbox_loc = gt_bbox_loc[mask, :]
        pos_mask = pos.unsqueeze(2).expand_as(pred_loc)
        # print(pos_mask.shape)
        pos_pred_locs = pred_loc[pos_mask].view(-1, 4)
        pos_loc_targets = gt_bbox_loc[pos_mask].view(-1, 4)

        loc_huber_loss = F.smooth_l1_loss(pos_pred_locs, pos_loc_targets, size_average=False)
        # print("loc_huber_loss")
        # print(loc_huber_loss.shape)
        # print('loc_huber_loss',loc_huber_loss)
        num_pos = num_pos.sum(dim=0).long().float()
        print('conf_loss:',conf_loss)
        print('loc_huber_loss',loc_huber_loss)
        conf_loss = conf_loss.sum(dim=0)
        loc_huber_loss = loc_huber_loss.sum(dim=0)
        conf_loss = conf_loss/num_pos
        loc_huber_loss = loc_huber_loss/num_pos

        return conf_loss, loc_huber_loss
