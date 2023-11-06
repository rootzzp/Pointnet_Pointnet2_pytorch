1 在data目录下新建custom目录
2 把解压后的txt文件夹放到custom目录中
3 创建 custom/pcds custom/test custom/trainval custom/edges目录
4 解注释掉data_utils/pre_process.py的第一步，把txt文件转换为pcd文件
5 注释掉一地步，解注释第二步，随机生成edge文件
6 运行train_custom.py进行训练