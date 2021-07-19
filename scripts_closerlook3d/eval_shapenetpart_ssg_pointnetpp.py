"""
Evaluate the trained model and compute iou.

Ask, what is iou? oh, my god.

The evaluation is proceeded in two-folds.
1. Semantic class level iou.
2. Instance level iou.
And therefore we need to compute the semantic confusion matrix for each object.
"""

from datasets.ShapeNetPart import ShapeNetPartSeg
import argparse
from torch.utils.data import DataLoader
from models.pointnet2_part_seg_ssg import get_model
import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

parser = argparse.ArgumentParser('ShapeNetPart part-segmentation training')
parser.add_argument('--data_root', type=str, default='data', metavar='PATH', help='root director of dataset')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--base_lr', type=float, default=1e-3, help='base learning rate')
parser.add_argument('--max_epoch', type=int, default=1000, help='number of training epochs')
parser.add_argument('--epochs_per_eval', type=int, default=10, help='number of training epochs')
parser.add_argument('--epochs_per_save', type=int, default=100, help='number of training epochs')
parser.add_argument('--experiment_name', type=str, default='output_seg_ssg_pointnetpp')
parser.add_argument('--checkpoint', type=str, default='200.pth')

args = parser.parse_args()

num_sem_class = 6
num_categories = 16

if __name__ == "__main__":
    train_dataset = ShapeNetPartSeg(num_points=2048, data_root=args.data_root, categories=['Table'], split='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = ShapeNetPartSeg(num_points=2048, data_root=args.data_root, categories=['Table'], split='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # no point cloud normal.
    net = get_model(num_classes=num_sem_class, normal_channel=False).cuda()
    net.load_state_dict(torch.load(os.path.join(args.experiment_name, args.checkpoint)))
    net.eval()

    # sucheng: we can use data augmentation and voting to cope with this problem.
    conf_mats = []
    for pcs, masks, point_labels, labels in tqdm(train_dataloader):
        onehot_labels = torch.zeros(size=(labels.shape[0], num_categories))
        onehot_labels = onehot_labels.scatter_(1, torch.unsqueeze(labels, dim=1), 1).cuda()

        pcs = pcs.cuda()
        pcs = pcs.transpose(1, 2)

        with torch.no_grad():
            pred_sem, global_feat = net(pcs, onehot_labels)

        batch_logits = []
        batch_points_labels = []
        batch_shape_labels = []
        batch_masks = []

        point_labels = point_labels.cuda()

        for batch_idx in range(pred_sem.shape[0]):
            this_pred_sem = pred_sem[batch_idx]
            this_point_labels = point_labels[batch_idx]

            this_pred_sem = torch.argmax(this_pred_sem, dim=1).cpu().numpy()
            this_point_labels = this_point_labels.cpu().numpy()
            conf_mat = confusion_matrix(this_point_labels, this_pred_sem, labels=np.arange(num_sem_class))
            conf_mats.append(conf_mat)

    # let us compute instance level of confusion matrix's mIOU.
    # therefore, we compute iou for each semantic class and take the average.
    # 1. true positive.
    # 2. the number of points that is labeled as some semantic class.
    # 3. the number of points that is predicted as some semantic class.
    ious = []
    for mat in conf_mats:
        tp = np.diagonal(mat)
        label = np.sum(mat, axis=1)
        pred = np.sum(mat, axis=0)
        iou = tp / (label + pred - tp + 1e-6)
        ious.append(iou)
    instance_miou = np.mean(np.stack(ious, axis=0), axis=0)

    # semantic class level mIOU.
    # why not represent it as a confusion matrix?
    conf_mat = np.sum(np.stack(conf_mats, axis=0), axis=0)
    conf_mat = conf_mat / (np.sum(conf_mat, axis=1, keepdims=True) + 1e-6)
    print('training set instance iou\t', instance_miou)
    print('training set semantic class conf mat\t', conf_mat)

    conf_mats = []
    for pcs, masks, point_labels, labels in tqdm(test_dataloader):
        onehot_labels = torch.zeros(size=(labels.shape[0], num_categories))
        onehot_labels = onehot_labels.scatter_(1, torch.unsqueeze(labels, dim=1), 1).cuda()

        pcs = pcs.cuda()
        pcs = pcs.transpose(1, 2)

        with torch.no_grad():
            pred_sem, global_feat = net(pcs, onehot_labels)

        batch_logits = []
        batch_points_labels = []
        batch_shape_labels = []
        batch_masks = []

        point_labels = point_labels.cuda()

        for batch_idx in range(pred_sem.shape[0]):
            this_pred_sem = pred_sem[batch_idx]
            this_point_labels = point_labels[batch_idx]

            this_pred_sem = torch.argmax(this_pred_sem, dim=1).cpu().numpy()
            this_point_labels = this_point_labels.cpu().numpy()
            conf_mat = confusion_matrix(this_point_labels, this_pred_sem, labels=np.arange(num_sem_class))
            conf_mats.append(conf_mat)

    # let us compute instance level of confusion matrix's mIOU.
    # therefore, we compute iou for each semantic class and take the average.
    # 1. true positive.
    # 2. the number of points that is labeled as some semantic class.
    # 3. the number of points that is predicted as some semantic class.
    ious = []
    for mat in conf_mats:
        tp = np.diagonal(mat)
        label = np.sum(mat, axis=1)
        pred = np.sum(mat, axis=0)
        iou = tp / (label + pred - tp + 1e-6)
        ious.append(np.mean(iou))
    instance_miou = np.mean(np.array(ious))

    # semantic class level mIOU.
    # why not represent it as a confusion matrix?
    conf_mat = np.sum(np.stack(conf_mats, axis=0), axis=0)
    conf_mat = conf_mat / (np.sum(conf_mat, axis=1, keepdims=True) + 1e-6)
    print('testing set instance iou\n', instance_miou)
    print('testing set semantic class conf mat\n', conf_mat)
