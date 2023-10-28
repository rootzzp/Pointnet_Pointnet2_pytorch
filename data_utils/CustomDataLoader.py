# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/custom', npoints=2500, split='train', test_ratio = 0.2, class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        file_list = os.listdir(root)
        file_list.remove("edge.txt")
        prefix_set = set()
        for n in file_list:
            prefix_set.add(n[0])
        pat_num = len(prefix_set)
        prefix_set = sorted(prefix_set)

        self.all = {}
        for part_name in prefix_set:
            tmp = []
            for n in file_list:
                if n.startswith(part_name):
                    tmp.append(n)
            tmp.sort(key = lambda x: int(x[1:-4]))
            self.all[part_name] = tmp

        self.part_labels = list(prefix_set)
        self.label2num = {}
        for i in range(len(self.part_labels)):
            self.label2num[self.part_labels[i]] = i + 1

        sizes = []
        keys = []
        for k,v in self.all.items():
            sizes.append(len(v))
            keys.append(k)
        self.max_num_part_index = np.argmax(sizes)
        self.max_part_label = keys[self.max_num_part_index]

        obj_num = sizes[self.max_num_part_index]
        train_num = int(obj_num * (1-test_ratio))
        test_num = obj_num - train_num

        self.datapath = []

        if split == 'trainval':
            for i in range(train_num):
                t = self.all[self.max_part_label][i]
                suffix = t[1:-4]
                self.process(suffix)
        elif split == 'test':
            for i in range(train_num, obj_num, 1):
                t = self.all[self.max_part_label][i]
                suffix = t[1:-4]
                self.process(suffix)

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def process(self,suffix,save_path="data/tmp"):
        save_file_path = os.path.join(save_path,"obj_"+suffix+".txt")
        self.datapath.append(save_file_path)
        if os.path.exists(save_file_path):
            return
        f = open(save_file_path, "w")
        for part_label in self.part_labels:
            part_file_name = part_label+suffix+".txt"
            label_n = self.label2num[part_label]
            part_file = os.path.join(self.root,part_file_name)
            if os.path.exists(part_file):
                file = open(part_file,'r')
                contents = file.readlines()
                for line in contents:
                    new_line = line.strip() + "\t" + str(label_n)
                    f.writelines(new_line)
                    f.writelines('\n')

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = 0
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)



