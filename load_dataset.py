import os
import joblib
import torch
from torch.utils.data import Dataset,Subset
from torch_geometric.loader import DataLoader
import numpy as np
import random
import pandas as pd
import os.path as osp
import h5py
from torch import Tensor
# import torch_geometric

def read_patch_feature(path: str, dtype:str='torch'):
    r"""Read node features from path.

    Args:
        dtype (string): Type of return data, default `torch`.
        path (string): Read data from path.
    """
    assert dtype in ['numpy', 'torch']
    ext = osp.splitext(path)[1]

    if ext == '.h5':
        with h5py.File(path, 'r') as hf:
            nfeats = hf['features'][:]
    elif ext == '.pt':
        nfeats = torch.load(path, map_location=torch.device('cpu'))
    else:
        raise ValueError(f'Not support {ext}')

    if isinstance(nfeats, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(nfeats)
    elif isinstance(nfeats, Tensor) and dtype == 'numpy':
        return nfeats.numpy()
    else:
        return nfeats


def read_patch_data(path: str, dtype:str='torch', key='features'):
    r"""Read patch data from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
        key (string): Key of return data, default 'features'.
    """
    assert dtype in ['numpy', 'torch']
    ext = osp.splitext(path)[1]

    if ext == '.h5':
        with h5py.File(path, 'r') as hf:
            pdata = hf[key][:]
    elif ext == '.pt':
        pdata = torch.load(path, map_location=torch.device('cpu'))
    elif ext == '.npy':
        pdata = np.load(path)
    else:
        raise ValueError(f'Not support {ext}')

    if isinstance(pdata, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(pdata)
    elif isinstance(pdata, Tensor) and dtype == 'numpy':
        return pdata.numpy()
    else:
        return pdata

def retrieve_from_table_clf(patient_ids, table_path, ret=None, level='slide', shuffle=False, 
    processing_table=None, pid_column='patient_id'):
    """Get info from table, oriented to classification tasks"""
    assert level in ['slide', 'patient']
    if ret is None:
        if level == 'patient':
            ret = ['pid', 'pid2sid', 'pid2label'] # for patient-level task
        else:
            ret = ['sid', 'sid2pid', 'sid2label'] # for slide-level task
    for r in ret:
        assert r in ['pid', 'sid', 'pid2sid', 'sid2pid', 'pid2label', 'sid2label']

    df = pd.read_csv(table_path, dtype={pid_column: str})
    assert_columns = [pid_column, 'pathology_id', 'label']
    for c in assert_columns:
        assert c in df.columns
    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    pid2loc = dict()
    for i in df.index:
        _p = df.loc[i, pid_column]
        if _p in patient_ids:
            if _p in pid2loc:
                pid2loc[_p].append(i)
            else:
                pid2loc[_p] = [i]

    pid, sid = list(), list()
    pid2sid, pid2label, sid2pid, sid2label = dict(), dict(), dict(), dict()
    for p in patient_ids:
        if p not in pid2loc:
            print('[Warning] Patient ID {} not found in table {}.'.format(p, table_path))
            continue
        pid.append(p)
        for _i in pid2loc[p]:
            _pid, _sid, _label = df.loc[_i, assert_columns].to_list()
            if _pid in pid2sid:
                pid2sid[_pid].append(_sid)
            else:
                pid2sid[_pid] = [_sid]
            if _pid not in pid2label:
                pid2label[_pid] = _label

            sid.append(_sid)
            sid2pid[_sid] = _pid
            sid2label[_sid] = _label

    if shuffle:
        if level == 'patient':
            pid = random.shuffle(pid)
        else:
            sid = random.shuffle(sid)

    res = []
    for r in ret:
        res.append(eval(r))
    return res

def retrieve_from_table_surv(patient_ids, table_path, ret=None, level='slide', shuffle=False, 
    processing_table=None, pid_column='patient_id'):
    """Get info from table, oriented to survival analysis tasks"""
    assert level in ['slide', 'patient']
    if ret is None:
        if level == 'patient':
            ret = ['pid', 'pid2sid', 'pid2label'] # for patient-level task
        else:
            ret = ['sid', 'sid2pid', 'sid2label'] # for slide-level task
    for r in ret:
        assert r in ['pid', 'sid', 'pid2sid', 'sid2pid', 'pid2label', 'sid2label']

    df = pd.read_csv(table_path, dtype={pid_column: str})
    assert_columns =  [pid_column, 'pathology_id', 'e', 't']
    for c in assert_columns:
        assert c in df.columns
    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    pid2loc = dict()
    for i in df.index:
        _p = df.loc[i, pid_column]
        if _p in patient_ids:
            if _p in pid2loc:
                pid2loc[_p].append(i)
            else:
                pid2loc[_p] = [i]

    pid, sid = list(), list()
    pid2sid, pid2label, sid2pid, sid2label = dict(), dict(), dict(), dict()
    for p in patient_ids:
        if p not in pid2loc:
            print('[Warning] Patient ID {} not found in table {}.'.format(p, table_path))
            continue
        pid.append(p)
        for _i in pid2loc[p]:
            _pid, _sid, _label,_rec, _pfs = df.loc[_i, assert_columns].to_list()
            if _pid in pid2sid:
                pid2sid[_pid].append(_sid)
            else:
                pid2sid[_pid] = [_sid]
            if _pid not in pid2label:
                pid2label[_pid] = {'label':_label,'rec': _rec, 'PFS': _pfs}

            sid.append(_sid)
            sid2pid[_sid] = _pid
            sid2label[_sid] ={'label':_label,'rec': _rec, 'PFS': _pfs}

    if shuffle:
        if level == 'patient':
            random.shuffle(pid)
        else:
            random.shuffle(sid)
    
    res = []
    for r in ret:
        res.append(eval(r))
    return res

def retrieve_from_table(patient_ids, table_path, ret=None, level='slide', shuffle=False, 
    processing_table=None, pid_column='patient_id', time_format='origin', time_bins=4):
    assert level in ['slide', 'patient']
    assert time_format in ['origin', 'ratio', 'quantile']
    if ret is None:
        if level == 'patient':
            ret = ['pid', 'pid2sid', 'pid2label'] # for patient-level task
        else:
            ret = ['sid', 'sid2pid', 'sid2label'] # for slide-level task
    for r in ret:
        assert r in ['pid', 'sid', 'pid2sid', 'sid2pid', 'pid2label', 'sid2label']

    df = pd.read_csv(table_path, dtype={pid_column: str})
    assert_columns = [pid_column, 'pathology_id', 't', 'e']
    for c in assert_columns:
        assert c in df.columns
    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    if shuffle:
        patient_ids = random.shuffle(patient_ids)

    pid2loc = dict()
    max_time = 0.0
    for i in df.index:
        max_time = max(max_time, df.loc[i, 't'])
        _p = df.loc[i, 'patient_id']
        if _p in patient_ids:
            if _p in pid2loc:
                pid2loc[_p].append(i)
            else:
                pid2loc[_p] = [i]

    # process time format
    if time_format == 'ratio':
        df.loc[:, 't'] = 1.0 * df.loc[:, 't'] / max_time
    elif time_format == 'quantile':
        df, new_columns = compute_discrete_label(df, bins=time_bins)
        assert_columns  = [pid_column, 'pathology_id'] + new_columns
    else:
        pass

    pid, sid = list(), list()
    pid2sid, pid2label, sid2pid, sid2label = dict(), dict(), dict(), dict()
    for p in patient_ids:
        if p not in pid2loc:
            print('[Warning] Patient ID {} not found in table {}.'.format(p, table_path))
        pid.append(p)
        for _i in pid2loc[p]:
            _pid, _sid, _t, _ind = df.loc[_i, assert_columns].to_list()
            if _pid in pid2sid:
                pid2sid[_pid].append(_sid)
            else:
                pid2sid[_pid] = [_sid]
            if _pid not in pid2label:
                pid2label[_pid] = (_t, _ind)

            sid.append(_sid)
            sid2pid[_sid] = _pid
            sid2label[_sid] = (_t, _ind)

    res = []
    for r in ret:
        res.append(eval(r))
    return res


def get_patient_data(df:pd.DataFrame, at_column='patient_id'):
    df_gps = df.groupby('patient_id').groups
    df_idx = [i[0] for i in df_gps.values()]
    pat_df = df.loc[df_idx, :]
    pat_df = pat_df.reset_index(drop=True)
    return pat_df

def compute_discrete_label(df:pd.DataFrame, column_t='t', column_e='e', bins=4):
    # merge all T and E
    min_t, max_t = df[column_t].min(), df[column_t].max()
    df.loc[:, 'y_c'] = 1 - df.loc[:, column_e] 

    # get patient data to generate their discrete time
    pat_df = get_patient_data(df)

    # qcut for patients
    df_evt    = pat_df[pat_df[column_e] == 1]
    _, qbins  = pd.qcut(df_evt[column_t], q=bins, retbins=True, labels=False)
    qbins[0]  = min_t - 1e-5
    qbins[-1] = max_t + 1e-5

    # cut for original data
    discrete_labels, qbins = pd.cut(df[column_t], bins=qbins, retbins=True, labels=False, right=False, include_lowest=True)
    df.loc[:, 'y_t'] = discrete_labels.values.astype(int)

    return df, ['y_t', 'y_c']




class CM_Dataset(Dataset):
    def __init__(self,fold_index,train_dir='/data12/zzf/MIL/CM16_256/train_patches/pkl_bak',test_dir='/data12/zzf/MIL/CM16_256/test_patches/pkl_bak'):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.test = 'test' in fold_index
        folds_file = joblib.load('/data12/zzf/MIL/CM16/data/fold_split/fold_split.pkl')
        if 'all' in fold_index:
            self.graphs = list(set(folds_file[fold_index.replace('all','train')])|set(folds_file[fold_index.replace('all','val')])|set(folds_file[fold_index.replace('all','test')]))
        else:
            self.graphs = folds_file[fold_index]
        self.num_features = 1024
        self.num_classes=2

    def __getitem__(self,index):
        if 'test' in self.graphs[index]:
            graph_path = os.path.join(self.test_dir,self.graphs[index])
        else:
            graph_path = os.path.join(self.train_dir,self.graphs[index])
        graph = torch.load(graph_path)
        return graph, graph_path
    
    def __len__(self):
        return len(self.graphs)

## 新建cm_dataset 
class CM_Dataset_tlz(Dataset):
    def __init__(self,fold_patient,train_dir='D:\\HIPT-GCN\\data\\20240704_downsample_2\\uni_v1\\pt_files_cluster',
                 table_path='D:/PseMix/data_split/crlm/converted_data.csv'):
        self.train_dir = train_dir        
      
        # folds_file = joblib.load('/data12/zzf/MIL/CM16/data/fold_split/fold_split.pkl')
        # if 'all' in fold_index:
        #     self.graphs = list(set(folds_file[fold_index.replace('all','train')])|set(folds_file[fold_index.replace('all','val')])|set(folds_file[fold_index.replace('all','test')]))
        # else:
        #     self.graphs = folds_file[fold_index]
        self.graphs = fold_patient
        info = ['sid', 'sid2pid', 'sid2label']
        self.sids, self.sid2pid, self.sid2label = retrieve_from_table_clf(
            fold_patient, table_path, ret=info, level='slide')
        self.num_features = 1024
        self.num_classes=2
        self.uid = self.sids
        self.flag_use_corrupted_label = False
        self.new_sid2label = None


    def __getitem__(self,index):
        # if 'test' in self.graphs[index]:
        #     graph_path = os.path.join(self.test_dir,self.graphs[index])
        # else:
        #     graph_path = os.path.join(self.train_dir,self.graphs[index])

        sid   = self.sids[index]
        pid   = self.sid2pid[sid]
        label = self.sid2label[sid] if not self.flag_use_corrupted_label else self.new_sid2label[sid]
        # get patches from one slide
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor([label]).to(torch.long)

        graph_path= os.path.join(self.train_dir,sid+'.pt')
        graph = torch.load(graph_path)
        return graph,label, graph_path
    
    def __len__(self):
        return len(self.graphs)

class CM_Dataset_tlz_surv(Dataset):
    def __init__(self,fold_patient,train_dir='D:\\HIPT-GCN\\data\\20240704_downsample_2\\uni_v1\\pt_files_cluster',
                 table_path='/home/daoyjz01/AdvMIL/table/tcga_brca_path_full.csv',time_format='ratio',time_bins=4,
                 read_format:str='pt',mode='cluster'):
        self.train_dir = train_dir        
        self.read_format= read_format
        # folds_file = joblib.load('/data12/zzf/MIL/CM16/data/fold_split/fold_split.pkl')
        # if 'all' in fold_index:
        #     self.graphs = list(set(folds_file[fold_index.replace('all','train')])|set(folds_file[fold_index.replace('all','val')])|set(folds_file[fold_index.replace('all','test')]))
        # else:
        #     self.graphs = folds_file[fold_index]
        self.graphs = fold_patient
        info = ['pid', 'pid2sid', 'pid2label']
        # self.sids, self.sid2pid, self.sid2label = retrieve_from_table_surv(
        #     fold_patient, table_path, ret=info, level='slide')        
        self.pids, self.pid2sid, self.pid2label = retrieve_from_table(
            fold_patient, table_path, ret=info, time_format=time_format, time_bins=time_bins)

        self.num_features = 1024
        self.num_classes=2
        self.uid = self.pids
        self.flag_use_corrupted_label = False
        self.new_sid2label = None
        self.mode = mode


    def __getitem__(self,index):
        # if 'test' in self.graphs[index]:
        #     graph_path = os.path.join(self.test_dir,self.graphs[index])
        # else:
        #     graph_path = os.path.join(self.train_dir,self.graphs[index])

        # sid   = self.pids[index]
        # pid   = self.sid2pid[sid]
        # label = self.sid2label[sid] if not self.flag_use_corrupted_label else self.new_sid2label[sid]
        pid   = self.pids[index]
        sids  = self.pid2sid[pid]
        label = self.pid2label[pid]
        # get patches from one slide
       
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)
        
        if self.mode == 'cluster':
            
            feats = []
            cluster_level1_10 = []
            cluster_level2_10 = []
            
            for sid in sids:
                full_path = osp.join(self.train_dir, sid + '.' + self.read_format)
                temp=torch.load(full_path)
                feats.append(temp['x'])
                cluster_level1_10.append(temp['cluster_level1_10'])
                cluster_level2_10.append(temp['cluster_level2_10'])
            feats = torch.cat(feats, dim=0).to(torch.float)
            cluster_level1_10 = torch.cat(cluster_level1_10, dim=0).to(torch.float)
            cluster_level2_10 = torch.cat(cluster_level2_10, dim=0).to(torch.float)
            
            tensor_structure = {
                'x': feats,
                'cluster_level1_10': cluster_level1_10,
                'cluster_level2_10': cluster_level2_10
            }

            # cids = torch.Tensor(cids)
            # assert cids.shape[0] == feats.shape[0]
            return tensor_structure,label, pid
        elif self.mode =='ABMIL'or self.mode=='DSMIL'or self.mode =='TransMIL' or self.mode=='CLAM':
            feats = []
            cluster_level1_10 = []
            cluster_level2_10 = []
            
            for sid in sids:
                full_path = osp.join(self.train_dir, sid + '.' + self.read_format)
                temp=torch.load(full_path)
                feats.append(temp['x'])
                cluster_level1_10.append(temp['cluster_level1_10'])
                cluster_level2_10.append(temp['cluster_level2_10'])
            feats = torch.cat(feats, dim=0).to(torch.float)
            cluster_level1_10 = torch.cat(cluster_level1_10, dim=0).to(torch.float)
            cluster_level2_10 = torch.cat(cluster_level2_10, dim=0).to(torch.float)
            
            tensor_structure = {
                'x': feats,
                'cluster_level1_10': cluster_level1_10,
                'cluster_level2_10': cluster_level2_10
            }

            # cids = torch.Tensor(cids)
            # assert cids.shape[0] == feats.shape[0]
            return tensor_structure,label, pid

        elif self.mode == 'graph':
            feats = []
            cluster_level1_10 = []
            cluster_level2_10 = []
            
            for sid in sids:
                full_path = osp.join(self.train_dir, sid + '.' + self.read_format)
                temp=torch.load(full_path)
                feats.append(temp['x'])
                cluster_level1_10.append(temp['cluster_level1_10'])
                cluster_level2_10.append(temp['cluster_level2_10'])
            feats = torch.cat(feats, dim=0).to(torch.float)
            cluster_level1_10 = torch.cat(cluster_level1_10, dim=0).to(torch.float)
            cluster_level2_10 = torch.cat(cluster_level2_10, dim=0).to(torch.float)
            
            tensor_structure = {
                'x': feats,
                'cluster_level1_10': cluster_level1_10,
                'cluster_level2_10': cluster_level2_10
            }

            graphs = []
            from .GraphBatchWSI import GraphBatch
            for sid in sids:
                # full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                # feats.append(read_patch_feature(full_path, dtype='torch'))
                full_graph = osp.join(self.kws['graph_path'],  sid + '.pt')
                graphs.append(torch.load(full_graph))
            # feats = torch.cat(feats, dim=0).to(torch.float)
            graphs = GraphBatch.from_data_list(graphs, update_cat_dims={'edge_latent': 1})
            assert isinstance(graphs, torch_geometric.data.Batch)
            return tensor_structure, label,graphs,pid  ## 这个地方返回得是四个元素


        return 
    
    def __len__(self):
        return len(self.graphs)
    

## psebmix的dataset
class CM_Dataset_tlz_psemix(Dataset):
    def __init__(self,fold_patient,train_dir='D:\\HIPT-GCN\\data\\20240704_downsample_2\\uni_v1\\pt_files_cluster',
                 table_path='/home/daoyjz01/AdvMIL/table/tcga_brca_path_full.csv',time_format='ratio',time_bins=4,
                 read_format:str='pt',mode='cluster'):
        self.train_dir = train_dir        
        self.read_format= read_format
        # folds_file = joblib.load('/data12/zzf/MIL/CM16/data/fold_split/fold_split.pkl')
        # if 'all' in fold_index:
        #     self.graphs = list(set(folds_file[fold_index.replace('all','train')])|set(folds_file[fold_index.replace('all','val')])|set(folds_file[fold_index.replace('all','test')]))
        # else:
        #     self.graphs = folds_file[fold_index]
        self.graphs = fold_patient
        info = ['pid', 'pid2sid', 'pid2label']
        # self.sids, self.sid2pid, self.sid2label = retrieve_from_table_surv(
        #     fold_patient, table_path, ret=info, level='slide')        
        self.pids, self.pid2sid, self.pid2label = retrieve_from_table(
            fold_patient, table_path, ret=info, time_format=time_format, time_bins=time_bins)

        self.num_features = 1024
        self.num_classes=2
        self.uid = self.pids
        self.flag_use_corrupted_label = False
        self.new_sid2label = None
        self.mode = mode


    def __getitem__(self,index):
        
        pid   = self.pids[index]
        sids  = self.pid2sid[pid]
        label = self.pid2label[pid]
        # get patches from one slide
       
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)
        
        if self.mode == 'cluster':
            
            feats = []
            cluster_level1_10 = []
            cluster_level2_10 = []
            
            for sid in sids:
                full_path = osp.join(self.train_dir, sid + '.' + self.read_format)
                temp=torch.load(full_path)
                feats.append(temp['x'])
                cluster_level1_10.append(temp['cluster_level1_10'])
                cluster_level2_10.append(temp['cluster_level2_10'])
            feats = torch.cat(feats, dim=0).to(torch.float)
            cluster_level1_10 = torch.cat(cluster_level1_10, dim=0).to(torch.float)
            cluster_level2_10 = torch.cat(cluster_level2_10, dim=0).to(torch.float)
            
            tensor_structure = {
                'x': feats,
                'cluster_level1_10': cluster_level1_10,
                'cluster_level2_10': cluster_level2_10
            }

            # cids = torch.Tensor(cids)
            # assert cids.shape[0] == feats.shape[0]
            return tensor_structure,label, (index,pid)

        elif self.mode =='ABMIL'or self.mode=='DSMIL'or self.mode =='TransMIL' or self.mode=='CLAM':
            feats = []
            cluster_level1_10 = []
            cluster_level2_10 = []
            
            for sid in sids:
                full_path = osp.join(self.train_dir, sid + '.' + self.read_format)
                temp=torch.load(full_path)
                feats.append(temp['x'])
                cluster_level1_10.append(temp['cluster_level1_10'])
                cluster_level2_10.append(temp['cluster_level2_10'])
            feats = torch.cat(feats, dim=0).to(torch.float)
            cluster_level1_10 = torch.cat(cluster_level1_10, dim=0).to(torch.float)
            cluster_level2_10 = torch.cat(cluster_level2_10, dim=0).to(torch.float)
            
            tensor_structure = {
                'x': feats,
                'cluster_level1_10': cluster_level1_10,
                'cluster_level2_10': cluster_level2_10
            }

            # cids = torch.Tensor(cids)
            # assert cids.shape[0] == feats.shape[0]
            return tensor_structure,label, (index,pid)


        return 
    
    def __len__(self):
        return len(self.graphs)


## remix 的dataset
class WSIProtoPatchSurv(Dataset):
    r"""A patch dataset class for classification tasks (slide-level in general).
    
    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self,fold_patient,train_dir='D:\\HIPT-GCN\\data\\20240704_downsample_2\\uni_v1\\pt_files_cluster',
                 table_path='/home/daoyjz01/AdvMIL/table/tcga_brca_path_full.csv',time_format='ratio',time_bins=4,
                 read_format:str='pt',mode='cluster'):
        super(WSIProtoPatchSurv, self).__init__()
        self.train_dir = train_dir        
        self.read_format= read_format
        # folds_file = joblib.load('/data12/zzf/MIL/CM16/data/fold_split/fold_split.pkl')
        # if 'all' in fold_index:
        #     self.graphs = list(set(folds_file[fold_index.replace('all','train')])|set(folds_file[fold_index.replace('all','val')])|set(folds_file[fold_index.replace('all','test')]))
        # else:
        #     self.graphs = folds_file[fold_index]
        self.graphs = fold_patient
        # info = ['pid', 'pid2sid', 'pid2label']
        # # self.sids, self.sid2pid, self.sid2label = retrieve_from_table_surv(
        # #     fold_patient, table_path, ret=info, level='slide')        
        # self.pids, self.pid2sid, self.pid2label = retrieve_from_table(
        #     fold_patient, table_path, ret=info, time_format=time_format, time_bins=time_bins)

        info = ['sid', 'sid2pid', 'sid2label']
        self.sids, self.sid2pid, self.sid2label = retrieve_from_table(
            fold_patient, table_path, ret=info, level='slide',time_format='ratio')

        self.num_features = 1024
        self.uid = self.sids

        self.mode = mode

    def __len__(self):
        return len(self.sids)
    def __getitem__(self,index):
        # if 'test' in self.graphs[index]:
        #     graph_path = os.path.join(self.test_dir,self.graphs[index])
        # else:
        #     graph_path = os.path.join(self.train_dir,self.graphs[index])

        # sid   = self.pids[index]
        # pid   = self.sid2pid[sid]
        # label = self.sid2label[sid] if not self.flag_use_corrupted_label else self.new_sid2label[sid]
        # pid   = self.pids[index]
        # sids  = self.pid2sid[pid]
        # label = self.pid2label[pid]
        sid   = self.sids[index]
        pid   = self.sid2pid[sid]
        label = self.sid2label[sid]
        # get patches from one slide
       
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)
        
      
        # load bag prototypes, i.e., reduced-bag
        cur_path = osp.join(self.train_dir, sid + '_bag_feats_proto_8.npy') # [n_cluster, dim_feat]
        feats = read_patch_data(cur_path, dtype='torch').to(torch.float)
        # load semantic shift vectors
        cur_path = os.path.join(self.train_dir, sid + '_bag_shift_proto_8.npy') # [n_cluster, n_shift_vector, dim_feat]
        semantic_shifts = read_patch_data(cur_path, dtype='torch').to(torch.float)
        # load patch labels (here only used as a placeholder)
        # patch_label = label * torch.ones(feats.shape[0]).to(torch.long)
        assert feats.shape[0] == semantic_shifts.shape[0]
        # return index, (feats, semantic_shifts), (label, patch_label)
        return (feats, semantic_shifts),label,pid
      

        # cids = torch.Tensor(cids)
        # assert cids.shape[0] == feats.shape[0]
        # return tensor_structure,label, pid
    
    def resume_labels(self):
        pass


def get_dataloader(index=0,seed=0):
    train_set=CM_Dataset(fold_index="train_{}".format(index))
    val_set=CM_Dataset(fold_index="val_{}".format(index))
    test_set=CM_Dataset(fold_index="test_{}".format(index))
    all_set=CM_Dataset(fold_index="all_{}".format(index))
    dataloader={}
    dataloader["train"]=DataLoader(train_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["all"]=DataLoader(all_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["val"]=DataLoader(val_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["test"]=DataLoader(test_set,batch_size=1,num_workers=0,drop_last=False)
   
    return dataloader


def read_datasplit_npz(path: str):
    data_npz = np.load(path)
    
    pids_train = [str(s) for s in data_npz['train_patients']]
    if 'val_patients' in data_npz:
        pids_val = [str(s) for s in data_npz['val_patients']]
    else:
        pids_val = None
    if 'test_patients' in data_npz:
        pids_test = [str(s) for s in data_npz['test_patients']]
    else:
        pids_test = None
    return pids_train, pids_val, pids_test

## 由于不知道文件结构，所以智能重写一个dataset
def get_dataloader_tlz(index=1,seed=0,):
    path_split='D:\\PseMix\\data_split\\crlm\\fold'+str(index)+'.npz'
    pids_train, pids_val, pids_test = read_datasplit_npz(path_split)

    train_set=CM_Dataset_tlz(pids_train)
    val_set=CM_Dataset_tlz(pids_val)
    test_set=CM_Dataset_tlz(pids_test)
    all_set=CM_Dataset_tlz(pids_train+pids_val+pids_test)
    dataloader={}
    dataloader["train"]=DataLoader(train_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["all"]=DataLoader(all_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["val"]=DataLoader(val_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["test"]=DataLoader(test_set,batch_size=1,num_workers=0,drop_last=False)
    # for graph, graph_path in dataloader["train"]:
    #     print(len(graph))
    return dataloader


def get_dataloader_tlz_remix(index=1,path_split='/home/daoyjz01/AdvMIL/data_split/tcga_brca-fold',
                            table_path = '/home/daoyjz01/AdvMIL/table/tcga_brca_path_full.csv',seed=0,train_dir=None,mode='cluster'):
    path_split=path_split+str(index)+'.npz'
    pids_train, pids_val, pids_test = read_datasplit_npz(path_split)

    train_set=WSIProtoPatchSurv(pids_train,train_dir=train_dir,mode=mode,table_path=table_path)
    val_set=WSIProtoPatchSurv(pids_val,train_dir=train_dir,mode=mode,table_path=table_path)
    test_set=WSIProtoPatchSurv(pids_test,train_dir=train_dir,mode=mode,table_path=table_path)
    all_set=WSIProtoPatchSurv(pids_train+pids_val+pids_test,train_dir=train_dir,mode=mode,table_path=table_path)
    dataloader={}
    dataloader["train"]=DataLoader(train_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["all"]=DataLoader(all_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["val"]=DataLoader(val_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["test"]=DataLoader(test_set,batch_size=1,num_workers=0,drop_last=False)
    # for graph, graph_path in dataloader["train"]:
    #     print(len(graph))
    return dataloader

def get_dataloader_tlz_surv(index=1,path_split='/home/daoyjz01/AdvMIL/data_split/tcga_brca-fold',
                            table_path = '/home/daoyjz01/AdvMIL/table/tcga_brca_path_full.csv',
                            seed=0,train_dir=None,mode='remix',train_ratio = 1.0):

    path_split=path_split+str(index)+'.npz'
    pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
    sample_size = int(len(pids_train) * train_ratio)
    pids_train = random.sample(pids_train, k=sample_size)


    train_set=CM_Dataset_tlz_surv(pids_train,train_dir=train_dir,mode=mode,table_path=table_path)
    val_set=CM_Dataset_tlz_surv(pids_val,train_dir=train_dir,mode=mode,table_path=table_path)
    test_set=CM_Dataset_tlz_surv(pids_test,train_dir=train_dir,mode=mode,table_path=table_path)
    all_set=CM_Dataset_tlz_surv(pids_train+pids_val+pids_test,train_dir=train_dir,mode=mode,table_path=table_path)
    dataloader={}
    dataloader["train"]=DataLoader(train_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["all"]=DataLoader(all_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["val"]=DataLoader(val_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["test"]=DataLoader(test_set,batch_size=1,num_workers=0,drop_last=False)
    # for graph, graph_path in dataloader["train"]:
    #     print(len(graph))
    return dataloader


def get_dataloader_tlz_psemix(index=1,path_split='/home/daoyjz01/AdvMIL/data_split/tcga_brca-fold',
                            table_path = '/home/daoyjz01/AdvMIL/table/tcga_brca_path_full.csv',seed=0,train_dir=None,mode='remix'):
    
    path_split=path_split+str(index)+'.npz'
    pids_train, pids_val, pids_test = read_datasplit_npz(path_split)

    train_set=CM_Dataset_tlz_psemix(pids_train,train_dir=train_dir,mode=mode,table_path=table_path)
    val_set=CM_Dataset_tlz_psemix(pids_val,train_dir=train_dir,mode=mode,table_path=table_path)
    test_set=CM_Dataset_tlz_psemix(pids_test,train_dir=train_dir,mode=mode,table_path=table_path)
    all_set=CM_Dataset_tlz_psemix(pids_train+pids_val+pids_test,train_dir=train_dir,mode=mode,table_path=table_path)
    dataloader={}
    dataloader["train"]=DataLoader(train_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["all"]=DataLoader(all_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["val"]=DataLoader(val_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["test"]=DataLoader(test_set,batch_size=1,num_workers=0,drop_last=False)
    # for graph, graph_path in dataloader["train"]:
    #     print(len(graph))
    return dataloader

# 测试
# get_dataloader_tlz()