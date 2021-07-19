from datasets.ShapeNetPart import ShapeNetPartSeg
import argparse
from torch.utils.data import DataLoader
from models.pointnet2_part_seg_ssg import get_model, get_loss
import torch
from tensorboardX import SummaryWriter
import os
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

args = parser.parse_args()

num_sem_class = 6
num_categories = 16

if __name__ == "__main__":
    train_dataset = ShapeNetPartSeg(num_points=2048, data_root=args.data_root, categories=['Table'], split='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = ShapeNetPartSeg(num_points=2048, data_root=args.data_root, categories=['Table'], split='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    # no point cloud normal.
    net = get_model(num_classes=num_sem_class, normal_channel=False).cuda()
    loss_fn = get_loss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200, gamma=0.5)

    writer = SummaryWriter(os.path.join(args.experiment_name, 'logs'))
    for epoch in tqdm(range(args.max_epoch)):

        net.train()
        epoch_loss = []
        for pcs, masks, point_labels, labels in train_dataloader:
            onehot_labels = torch.zeros(size=(labels.shape[0], num_categories))
            onehot_labels = onehot_labels.scatter_(1, torch.unsqueeze(labels, dim=1), 1).cuda()

            pcs = pcs.cuda()
            pcs = pcs.transpose(1, 2)
            pred_sem, global_feat = net(pcs, onehot_labels)

            point_labels = point_labels.cuda()

            pred_sem = torch.reshape(pred_sem, shape=(-1, num_sem_class)).contiguous()
            point_labels = torch.reshape(point_labels, shape=(-1,)).contiguous()
            # loss = torch.sum(pred_sem)
            loss = loss_fn(pred_sem, point_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        writer.add_scalar('train_loss', sum(epoch_loss) / len(epoch_loss), global_step=epoch)

        scheduler.step()

        if epoch % args.epochs_per_eval == 0:
            net.eval()
            epoch_loss = []
            for pcs, masks, point_labels, labels in val_dataloader:
                onehot_labels = torch.zeros(size=(labels.shape[0], num_categories))
                onehot_labels = onehot_labels.scatter_(1, torch.unsqueeze(labels, dim=1), 1).cuda()

                pcs = pcs.cuda()
                pcs = pcs.transpose(1, 2)
                with torch.no_grad():
                    pred_sem, global_feat = net(pcs, onehot_labels)

                point_labels = point_labels.cuda()

                loss = loss_fn(torch.reshape(pred_sem, shape=(-1, num_sem_class)),
                               torch.reshape(point_labels, shape=(-1,)))

                epoch_loss.append(loss.item())
            writer.add_scalar('val_loss', sum(epoch_loss) / len(epoch_loss), global_step=epoch)

        if epoch % args.epochs_per_save == 0:
            torch.save(net.state_dict(), os.path.join(args.experiment_name, f'{epoch}.pth'))
