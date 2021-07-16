from datasets.ShapeNetPart import ShapeNetPartSeg
import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser('ShapeNetPart part-segmentation training')
parser.add_argument('--data_root', type=str, default='data', metavar='PATH', help='root director of dataset')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--base_learning_rate', type=float, default=1e-3, help='base learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

args = parser.parse_args()

if __name__ == "__main__":
    dataset = ShapeNetPartSeg(num_points=4096, data_root=args.data_root)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

