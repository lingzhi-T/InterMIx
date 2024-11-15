import warnings
warnings.filterwarnings("ignore")
import os 
import sys
import yaml
import torch 
import numpy as np 
from tqdm import tqdm 
from sklearn.cluster import KMeans
from easydict import EasyDict
## 20240723 自己的auto生成的patch做数据集做两层聚类

train_pyg_dir = "/svs_file/bingli_20240729/brca_processed/feat-l1-uni_v1-B/pt_files"
ks = [10,5,4]
seed = 0
store_train_dir = 'svs_file/bingli_20240729/brca_processed/feat-l1-uni_v1-B/pt_files_cluster'
if not os.path.exists(store_train_dir):
    os.makedirs(store_train_dir)
for i in [ks[0]]:
    c0 = KMeans(n_clusters=i, random_state=seed)
    c = KMeans(n_clusters=ks[1], random_state=seed)
    for pkl in tqdm(os.listdir(train_pyg_dir)):
        print(pkl)        
        loaded_object = torch.load(os.path.join(train_pyg_dir, pkl), map_location=torch.device('cpu'))        
        if isinstance(loaded_object, dict):
            g = loaded_object
        else:
            print(f"警告：文件 {pkl} 不包含字典，使用空字典。")
            g = {}
            g['x'] = torch.load(os.path.join(train_pyg_dir, pkl), map_location=torch.device('cpu'))

        # g = torch.load(os.path.join(train_pyg_dir,pkl),map_location=torch.device('cpu'))
        cluster = c0.fit(g['x'].cpu().numpy())
        g['cluster_level1_'+str(i)] = torch.LongTensor(cluster.labels_).cpu()
        g['cluster_level2_'+str(i)] = -1 * torch.ones(len(g['x'])).cpu().long()
        for ci in range(i):
            if ks[1] > len(g['x'][g['cluster_level1_'+str(i)]==ci]):
                c_i = KMeans(n_clusters=len(g['x'][g['cluster_level1_'+str(i)]==ci]), random_state=0)
                cluster = c_i.fit(g['x'][g['cluster_level1_'+str(i)]==ci])
            else:
                cluster = c.fit(g['x'][g['cluster_level1_'+str(i)]==ci])
            g['cluster_level2_'+str(i)][g['cluster_level1_'+str(i)]==ci] = torch.LongTensor(cluster.labels_).cpu()
        torch.save(g,os.path.join(store_train_dir,pkl))
        print(np.unique(cluster.labels_,return_counts=True))