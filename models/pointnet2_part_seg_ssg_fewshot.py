import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6 + additional_channel,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 6 + additional_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, sx, sy, qx, num_parts):
        """
        here we conduct few-shot segmentation.
        sucheng: do a simple prototypical semantic segmentation here.
        """
        # do per-point feature extraction.
        B, Shot, N, C = sx.shape

        sx = torch.reshape(sx, shape=(B * Shot, N, C))
        sx = sx.transpose(1, 2)
        qx = qx.transpose(1, 2)

        # use Siamese network to get per-point deep feature.
        l0_points = sx
        l0_xyz = sx
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        sx_feat = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        sx_feat = torch.reshape(sx_feat, shape=(B, Shot, -1, N))

        l0_points = qx
        l0_xyz = qx
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        qx_feat = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # compute embedding for each part.
        # for each query, compute embedding separately for its associate support shape.
        pred_qy = []
        for b in range(B):
            # iterate through part feature.
            part_feat = [[] for _ in range(num_parts)]
            for p in range(num_parts):
                # iterate through each support datum.
                for s in range(Shot):
                    # index this part.
                    part_idx = torch.squeeze(torch.nonzero(sy[b, s] == p))
                    if len(part_idx.shape) > 0 and part_idx.shape[0] > 0:
                        sx_part_feat = sx_feat[b, s][:, part_idx]
                        # do an average pooling over point feature.
                        part_feat[p].append(torch.mean(sx_part_feat, dim=1))
                # average pooling over shots.
                if len(part_feat[p]) > 0:
                    part_feat[p] = torch.mean(torch.stack(part_feat[p], dim=0), dim=0)
                else:
                    # if there's no corresponding semantic part in the reference shape, use a default one.
                    part_feat[p] = torch.zeros_like(sx_feat[0, 0, :, 0], requires_grad=False)

            # for each query, do prototypical semantic segmentation for each point in the queried shape.
            # shape (num_parts, point_feat_dim).
            prototypes = torch.stack(part_feat, dim=0)
            # shape (point_feat_dim, num_points).
            this_qx_feat = qx_feat[b]

            # broadcast shape to (num_parts, num_points, point_feat_dim).
            num_parts, feat_dim, num_points = prototypes.shape[0], prototypes.shape[1], this_qx_feat.shape[1]
            prototypes = torch.unsqueeze(prototypes, dim=1).repeat(1, num_points, 1)
            this_qx_feat = torch.unsqueeze(this_qx_feat.transpose(0, 1), dim=0).repeat(num_parts, 1, 1)

            # compute euclidean distance since it is used in original prototypical networks paper.
            euc_dist = torch.mean((prototypes - this_qx_feat) ** 2, dim=2)
            this_pred_qy = F.softmax(euc_dist, dim=0)
            pred_qy.append(this_pred_qy.transpose(0, 1))

        # shape (batch_size, num_points, sem_label).
        pred_qy = torch.stack(pred_qy, dim=0)
        return pred_qy


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
