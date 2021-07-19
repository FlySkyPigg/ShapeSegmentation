"""
This piece of code comes from CloserLook3D.
"""

import os
import torch
import json
import shlex
import pickle
import subprocess
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from pytorch3d.loss import chamfer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def pc_normalize(pc):
    # Center and rescale point for 1m radius
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale

    return pc


class ShapeNetPartSeg(data.Dataset):
    def __init__(self, num_points, data_root=None, transforms=None, split='train', download=True, categories=None):
        self.transforms = transforms
        self.num_points = num_points
        self.split = split
        self.label_to_names = {0: 'Airplane',
                               1: 'Bag',
                               2: 'Cap',
                               3: 'Car',
                               4: 'Chair',
                               5: 'Earphone',
                               6: 'Guitar',
                               7: 'Knife',
                               8: 'Lamp',
                               9: 'Laptop',
                               10: 'Motorbike',
                               11: 'Mug',
                               12: 'Pistol',
                               13: 'Rocket',
                               14: 'Skateboard',
                               15: 'Table'}

        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.num_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]

        self.data_root = data_root
        if data_root is None:
            self.data_root = os.path.join(ROOT_DIR, 'data')
        else:
            self.data_root = data_root
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.folder = "ShapeNetPart"
        self.data_dir = os.path.join(self.data_root, self.folder, 'shapenetcore_partanno_segmentation_benchmark_v0')
        self.url = "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(self.data_root, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, self.data_root))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.category_and_synsetoffset = [['Airplane', '02691156'],
                                          ['Bag', '02773838'],
                                          ['Cap', '02954340'],
                                          ['Car', '02958343'],
                                          ['Chair', '03001627'],
                                          ['Earphone', '03261776'],
                                          ['Guitar', '03467517'],
                                          ['Knife', '03624134'],
                                          ['Lamp', '03636649'],
                                          ['Laptop', '03642806'],
                                          ['Motorbike', '03790512'],
                                          ['Mug', '03797390'],
                                          ['Pistol', '03948459'],
                                          ['Rocket', '04099429'],
                                          ['Skateboard', '04225987'],
                                          ['Table', '04379243']]
        synsetoffset_to_category = {s: n for n, s in self.category_and_synsetoffset}

        # Train split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_train_file_list.json')
        with open(split_file, 'r') as f:
            train_files = json.load(f)
        train_files = [name[11:] for name in train_files]

        # Val split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_val_file_list.json')
        with open(split_file, 'r') as f:
            val_files = json.load(f)
        val_files = [name[11:] for name in val_files]

        # Test split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_test_file_list.json')
        with open(split_file, 'r') as f:
            test_files = json.load(f)
        test_files = [name[11:] for name in test_files]

        split_files = {'train': train_files,
                       'trainval': train_files + val_files,
                       'val': val_files,
                       'test': test_files
                       }
        files = split_files[split]
        # sucheng: plan to build a cache here.
        filename = os.path.join(self.data_root, self.folder, '{}_data.pkl'.format(split))
        from tqdm import tqdm
        if not os.path.exists(filename):
            point_list = []
            points_label_list = []
            label_list = []
            for i, file in tqdm(enumerate(files)):
                # Get class
                synset = file.split('/')[0]
                class_name = synsetoffset_to_category[synset]
                cls = self.name_to_label[class_name]
                cls = np.array(cls)
                # Get filename
                file_name = file.split('/')[1]
                # Load points and labels
                point_set = np.loadtxt(os.path.join(self.data_dir, synset, 'points', file_name + '.pts')).astype(
                    np.float32)
                point_set = pc_normalize(point_set)
                seg = np.loadtxt(os.path.join(self.data_dir, synset, 'points_label', file_name + '.seg')).astype(
                    np.int64) - 1
                point_list.append(point_set)
                points_label_list.append(seg)
                label_list.append(cls)
            self.points = point_list
            self.point_labels = points_label_list
            self.labels = label_list
            with open(filename, 'wb') as f:
                pickle.dump((self.points, self.point_labels, self.labels), f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.points, self.point_labels, self.labels = pickle.load(f)
            print(f"{filename} loaded successfully")

        # sucheng: filter data given category id.
        cat_labels = [self.name_to_label[cat_name] for cat_name in categories]
        indices = [i for i, label in enumerate(self.labels) if label in cat_labels]
        self.points = [self.points[idx] for idx in indices]
        self.point_labels = [self.point_labels[idx] for idx in indices]
        self.labels = [self.labels[idx] for idx in indices]

        print(f"split:{split} had {len(self.points)} data")

    def __getitem__(self, idx):
        current_points = self.points[idx]
        current_point_labels = self.point_labels[idx]
        cur_num_points = current_points.shape[0]
        if cur_num_points >= self.num_points:
            choice = np.random.choice(cur_num_points, self.num_points)
            current_points = current_points[choice, :]
            current_point_labels = current_point_labels[choice]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            padding_num = self.num_points - cur_num_points
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            padding_choice = np.random.choice(cur_num_points, padding_num)
            choice = np.hstack([shuffle_choice, padding_choice])
            current_points = current_points[choice, :]
            current_point_labels = current_point_labels[choice]
            mask = torch.cat([torch.ones(cur_num_points), torch.zeros(padding_num)]).type(torch.int32)

        label = torch.from_numpy(self.labels[idx]).type(torch.int64)
        current_point_labels = torch.from_numpy(current_point_labels).type(torch.int64)
        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, mask, current_point_labels, label

    def __len__(self):
        return len(self.points)


class FewShotShapeNetPart(data.Dataset):
    """
    Few-shot learning dataset:
        For each shape class, -> C-way K-shot.
            where C is the number of semantic categories in the shape class,
            and K is the support shapes in the same class (randomly sample for now).
        All support data comes from training dataset, and the query data comes from the specified split dataset.
    This code is adapted from ShapeNetPartSeg dataset.
    """

    def __init__(self, num_points, data_root=None, transforms=None, split='train', category=None, K: int = 5):
        # sucheng: assume the cache has been processed.
        #   only support one category.
        super(FewShotShapeNetPart, self).__init__()

        self.transforms = transforms
        self.num_points = num_points
        self.split = split
        self.category = category
        self.label_to_names = {0: 'Airplane',
                               1: 'Bag',
                               2: 'Cap',
                               3: 'Car',
                               4: 'Chair',
                               5: 'Earphone',
                               6: 'Guitar',
                               7: 'Knife',
                               8: 'Lamp',
                               9: 'Laptop',
                               10: 'Motorbike',
                               11: 'Mug',
                               12: 'Pistol',
                               13: 'Rocket',
                               14: 'Skateboard',
                               15: 'Table'}

        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self._num_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]

        self.data_root = data_root
        if data_root is None:
            self.data_root = os.path.join(ROOT_DIR, 'data')
        else:
            self.data_root = data_root
        assert os.path.exists(self.data_root)

        self.folder = "ShapeNetPart"

        with open(os.path.join(self.data_root, self.folder, 'train_data.pkl'), 'rb') as f:
            self.train_points, self.train_point_labels, self.train_labels = pickle.load(f)

        with open(os.path.join(self.data_root, self.folder, f'{split}_data.pkl'), 'rb') as f:
            self.split_points, self.split_point_labels, self.split_labels = pickle.load(f)

        # sucheng: filter data by the selected categories, and by point number.
        cat_label = self.name_to_label[self.category]
        indices = [i for i, label in enumerate(self.train_labels) if label == cat_label and self.train_points[i].shape[0] >= self.num_points]
        self.train_points = [self.train_points[idx] for idx in indices]
        self.train_point_labels = [self.train_point_labels[idx] for idx in indices]

        indices = [i for i, label in enumerate(self.split_labels) if label == cat_label and self.split_points[i].shape[0] >= self.num_points]
        self.split_points = [self.split_points[idx] for idx in indices]
        self.split_point_labels = [self.split_point_labels[idx] for idx in indices]


        # sucheng: K shot.
        self.K = K
        # sucheng: load knn index file for few-shot support set retrieval.
        retrieval_index_filename = os.path.join(self.data_root, self.folder, f'{self.split}_retrieval_index.npy')
        if not os.path.exists(retrieval_index_filename):
            self._build_knn_indices()
        self.retrieval_index = np.load(retrieval_index_filename)

    @property
    def num_parts(self):
        return self._num_parts[self.name_to_label[self.category]]

    def _build_knn_indices(self):
        # sucheng: build a knn index file for few-shot support set choice.
        #   for each datum in this split, query for K nearest neighbor in training set.
        knn_indices = []

        for i in range(len(self)):
            query_points = self.split_points[i]
            query_point_labels = self.split_point_labels[i]
            query_part_points = [query_points[query_point_labels == sem_id] for sem_id in range(self.num_parts)]

            # iterate over each training sample.
            chamfer_dists = []
            for j in tqdm(range(len(self.train_points))):
                if j == 100:
                    break
                support_points = self.train_points[j]
                support_point_labels = self.train_point_labels[j]
                support_part_points = [support_points[support_point_labels == sem_id] for sem_id in range(self.num_parts)]

                # iterate over each part.
                chamfer_dist = 0
                for k in range(self.num_parts):
                    query_part, support_part = query_part_points[k], support_part_points[k]
                    query_part, support_part = torch.from_numpy(query_part).float(), torch.from_numpy(
                        support_part).float()

                    if query_part.shape[0] == support_part.shape[0] == 0:
                        dist = 0
                    elif query_part.shape[0] == 0 or support_part.shape[0] == 0:
                        dist = 1000
                    else:
                        dist, _ = chamfer.chamfer_distance(torch.unsqueeze(query_part, dim=0),
                                                           torch.unsqueeze(support_part, dim=0))
                    assert isinstance(dist, int) or not torch.isnan(dist)
                    chamfer_dist = chamfer_dist + dist
                chamfer_dist /= self.num_parts
                chamfer_dists.append(chamfer_dist)

            chamfer_dists = np.array(chamfer_dists)
            neighbor_idx = np.argsort(chamfer_dists)[:50]
            knn_indices.append(neighbor_idx)

        knn_indices = np.stack(knn_indices, axis=0)
        retrieval_index_filename = os.path.join(self.data_root, self.folder, f'{self.split}_retrieval_index.npy')
        np.save(retrieval_index_filename, knn_indices)

    def __getitem__(self, idx):
        """
        C-way K-shot support set.
        The support set is randomly sampled from
        """
        # sucheng: support set is sampled from training shapes.
        support_indices = self.retrieval_index[idx][1:self.K + 1]  # skip the most similar support sample.
        support_points = [self.train_points[i] for i in support_indices]
        support_point_labels = [self.train_point_labels[i] for i in support_indices]

        query_points = self.split_points[idx]
        query_point_labels = self.split_point_labels[idx]

        for i in range(len(support_points)):
            rand_idx = np.random.choice(np.arange(support_points[i].shape[0]), size=(self.num_points,), replace=False)
            support_points[i] = support_points[i][rand_idx]
            support_point_labels[i] = support_point_labels[i][rand_idx]

        rand_idx = np.random.choice(np.arange(query_points.shape[0]), size=(self.num_points,), replace=False)
        query_points = query_points[rand_idx]
        query_point_labels = query_point_labels[rand_idx]

        support_points = torch.from_numpy(np.stack(support_points, axis=0)).float()
        support_point_labels = torch.from_numpy(np.stack(support_point_labels, axis=0)).float()
        query_points = torch.from_numpy(query_points).float()
        query_point_labels = torch.from_numpy(query_point_labels).long()

        return support_points, support_point_labels, query_points, query_point_labels

    def __len__(self):
        return 10
        return len(self.split_points)
