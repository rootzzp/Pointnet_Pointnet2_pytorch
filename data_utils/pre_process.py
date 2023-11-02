import numpy as np
import os
import random

label_to_color = {1:(255 << 16 | 0 << 8 | 0),
0:(0 << 16 | 255 << 8 | 0)}
color_to_label = {(255 << 16 | 0 << 8 | 0):1,
(0 << 16 | 255 << 8 | 0):0}

def txt_to_pcd(txt_file, pcd_file):
    # 打开txt文件并读取数据
    # with open(txt_file, 'r') as f:
    f = open(txt_file, 'r')
    lines = f.readlines()
    label = 1 if "edge" in txt_file else 0
    # 创建新的pcd文件并写入pcd文件头部信息
    with open(pcd_file, 'w') as f:
        # 写入pcd文件头部信息
        f.write("VERSION .7\n")
        f.write("FIELDS x y z normal_x normal_y normal_z rgba\n")
        f.write("SIZE 4 4 4 4 4 4 4\n")
        f.write("TYPE F F F F F F U\n")
        f.write("COUNT 1 1 1 1 1 1 1\n")
        f.write("WIDTH {}\n".format(len(lines)))
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0 0\n")
        f.write("POINTS {}\n".format(len(lines)))
        f.write("DATA ascii\n")

        # 逐行读取txt文件中的数据并写入pcd文件
        for line in lines:
            data = line.strip().split('\t')
            x, y, z, nx, ny, nz = map(float, data)
            f.write("{} {} {} {} {} {} {}\n".format(x, y, z, nx, ny, nz, label_to_color[label]))

    print("转换完成！")
    f.close()

def points_to_pcd(points, pcd_file, with_label = False):
    # 创建新的pcd文件并写入pcd文件头部信息
    size = len(points)
    with open(pcd_file, 'w') as f:
        # 写入pcd文件头部信息
        f.write("VERSION .7\n")
        if with_label:
            f.write("FIELDS x y z normal_x normal_y normal_z rgba\n")
            f.write("SIZE 4 4 4 4 4 4 4\n")
            f.write("TYPE F F F F F F U\n")
            f.write("COUNT 1 1 1 1 1 1 1\n")
            f.write("WIDTH {}\n".format(size))
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0 0\n")
        else:
            f.write("FIELDS x y z normal_x normal_y normal_z\n")
            f.write("SIZE 4 4 4 4 4 4\n")
            f.write("TYPE F F F F F F\n")
            f.write("COUNT 1 1 1 1 1 1\n")
            f.write("WIDTH {}\n".format(size))
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS {}\n".format(size))
        f.write("DATA ascii\n")
        if with_label:
            for point in points:
                x, y, z, nx, ny, nz,l = point
                f.write("{} {} {} {} {} {} {}\n".format(x, y, z, nx, ny, nz, l))
        else:
            for point in points:
                x, y, z, nx, ny, nz = point
                f.write("{} {} {} {} {} {}\n".format(x, y, z, nx, ny, nz))

    print("转换完成！")
    f.close()

def read_ascii_points(file):
    points = []
    for line in file:
        point = [float(value) for value in line.strip().split()]
        points.append(point)

    return points

def read_pcd_file(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件头部，提取点云数据相关信息
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line.startswith('DATA'):
                break
        points = read_ascii_points(f)

    return np.asarray(points)

def generate_edge(path,save_fold='./data/custom/edges/',min_ratio = 0.05, max_ratio = 1.0, count = 500):
    points = read_pcd_file(path)
    size = points.shape[0]

    indices = list(range(size))

    if not os.path.exists(save_fold):
        os.makedirs(save_fold)

    while count:
        random.shuffle(indices)
        count = count - 1
        ratio = random.uniform(min_ratio,max_ratio)
        # 计算采样点数
        n_points = int(size * ratio)

        # 随机选择采样点
        start_index = np.random.randint(size - n_points)
        idx = [*range(start_index,start_index+n_points)]

        # 选取采样点
        sampled_pcd_array = points[idx]
        save_path = os.path.join(save_fold,"edge_"+str(count)+".pcd")
        points_to_pcd(sampled_pcd_array,save_path,True)
        down_sample_size = len(sampled_pcd_array)
        print("edge has point {} after down sample have {} points".format(size,down_sample_size))


def get_down_sample_edge(ratio=0.1):
    file = "edge.pcd"
    path = os.path.join('./data/pcds',file)
    points = read_pcd_file(path)
    size = points.shape[0]

    sample_size = int(ratio*size)

    indices = list(range(size))
    random.shuffle(indices)
    idx = indices[:sample_size]
    sampled_pcd_array = points[idx]
    save_path = os.path.join('./data/',"downsample_edge.pcd")
    points_to_pcd(sampled_pcd_array,save_path,True)

if __name__ == "__main__":

    # first
    # files = os.listdir('./data/custom/txt/')
    # for file in files:
    #     path = os.path.join('./data/custom/txt/',file)
    #     save_path = os.path.join('./data/custom/pcds/',file[:-4]+".pcd")
    #     txt_to_pcd(path,save_path)

    # second
    edge_file = os.path.join('./data/custom/pcds',"edge.pcd")
    save_fold = './data/custom/edges'
    generate_edge(edge_file,save_fold=save_fold,count=1000)

    # maybe
    # get_down_sample_edge()
    # generate_edge(os.path.join('./data/',"downsample_edge.pcd"), './data/downsample_edges/')

    # root = 'data/pcds/'
    # edge = './data/downsample_edges'
    # generate_data(root,edge,600,100)