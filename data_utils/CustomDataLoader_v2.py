# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import random
from .pre_process import read_pcd_file, points_to_pcd, label_to_color, color_to_label


def generate_data(root = './data/custom/pcds',
                  edge_fold = './data/custom/edges',
                  train_fold_name = 'trainval',
                  test_fold_name = 'test',
                  generate_train_count = 2000,generate_test_count = 400,random_size = 10):
    parent_dir, _ = os.path.split(os.path.dirname(root))
    trainval_save_fold = os.path.join(parent_dir,train_fold_name)
    test_save_fold = os.path.join(parent_dir,test_fold_name)
    if not os.path.exists(trainval_save_fold):
        os.makedirs(trainval_save_fold)
    if not os.path.exists(test_save_fold):
        os.makedirs(test_save_fold)

    file_list = os.listdir(root)
    file_list.remove("edge.pcd")
    size = len(file_list)

    file_list = [os.path.join(root,a) for a in file_list]

    edge_files = os.listdir(edge_fold)
    edge_files = [os.path.join(edge_fold, a) for a in edge_files]

    train_sequences = []
    test_sequences = []

    indices = list(range(len(file_list)))

    # 随机选择训练集的子序列
    while len(train_sequences) < generate_train_count:
        random.shuffle(indices)
        train_indices = sorted(indices[:random.randint(1,size-1)])
        edge_file = edge_files[random.randint(0,len(edge_files)-1)]
        train_sequences.append([file_list[i] for i in train_indices] + [edge_file])

    # 随机选择测试集的子序列
    while len(test_sequences) < generate_test_count:
        random.shuffle(indices)
        test_indices = sorted(indices[:random.randint(1,size-1)])
        edge_file = edge_files[random.randint(0,len(edge_files)-1)]
        test_sequences.append([file_list[i] for i in test_indices] + [edge_file])

    for i,files in enumerate(train_sequences):
        save_file_path = os.path.join(trainval_save_fold,"train_obj_"+str(i)+".pcd")
        if os.path.exists(save_file_path):
            continue
        concat_points = []
        for part_file in files:
            points = read_pcd_file(part_file)
            concat_points.append(points)
        concat_points = np.concatenate(tuple(concat_points))
        points_to_pcd(concat_points,save_file_path,True)

    for i,files in enumerate(test_sequences):
        save_file_path = os.path.join(test_save_fold,"test_obj_"+str(i)+".pcd")
        if os.path.exists(save_file_path):
            continue
        concat_points = []
        for part_file in files:
            points = read_pcd_file(part_file)
            concat_points.append(points)
        concat_points = np.concatenate(tuple(concat_points))
        points_to_pcd(concat_points,save_file_path,True)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/custom/pcds', npoints=2500, split='train', generate_count = 0.2, class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel

        parent_dir, _ = os.path.split(os.path.dirname(root))
        self.save_fold = os.path.join(parent_dir,split)
        self.datapath = os.listdir(self.save_fold)
        self.datapath = [os.path.join(self.save_fold, a) for a in self.datapath]

        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        
        fn = self.datapath[index]
        cls = 0
        cls = np.array([cls]).astype(np.int32)
        data = read_pcd_file(fn)
        if not self.normal_channel:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:6]
        seg = [color_to_label[a] for a in data[:, -1]]
        seg = np.array(seg,dtype=np.int32)
        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)