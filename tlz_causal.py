## 这里只包括引用的文件
import os
import sys 
import math
import torch
import joblib
import random 
import yaml

import numpy as np 
from tqdm import tqdm 

from sklearn.cluster import KMeans
from load_dataset import get_dataloader,get_dataloader_tlz,get_dataloader_tlz_surv
from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from tlz_utils.core import PseudoBag_tlz


def evaluate_stability(y_hat, Stable_threshold):
    """
    计算均值和方差并评估稳定性。
    """
    mean = sum(y_hat) / len(y_hat)
    variance = sum((x - mean) ** 2 for x in y_hat) / (len(y_hat) - 1)
    return variance < Stable_threshold, mean, variance

def generate_pseudo_bag(data_x, data_cluster_level1, pseudo_bag, milnet, ratios,mode,cishu=10):
    """
    生成伪袋并计算10次采样后的模型输出。
    """
    y_hat = []
    for i in range(cishu):
        selected_ratio = random.choice(ratios)
        x_aug,x_aug_cluster= pseudo_bag.divide_20240724(data_x.squeeze(), data_cluster_level1, sample_proportion=selected_ratio)
        if mode =='cluster':  ## 特殊情况 对于一些模型输入需要cluster信息
            logit_label = milnet(x_aug,x_aug_cluster)
        else:
            logit_label = milnet(x_aug)
        y_hat.append(logit_label)
    return y_hat

def assess_risk(one_cluster_risk_duo, one_cluster_risk_shao, high_risk_ratio, low_risk_ratio, high_ratio, low_ratio, confounding_ratio, high_risk, low_risk, confounding, j, proportions_new):
    """
    风险评估
    """
    if one_cluster_risk_duo < high_risk_ratio and one_cluster_risk_shao < high_risk_ratio:
        high_risk.append(j)
        high_ratio += proportions_new[j]
    elif one_cluster_risk_duo > low_risk_ratio and one_cluster_risk_shao > low_risk_ratio:
        low_risk.append(j)
        low_ratio += proportions_new[j]
    else:
        confounding.append(j)
        confounding_ratio += proportions_new[j]
    return high_ratio, low_ratio, confounding_ratio

def assess_stability(one_cluster_risk_duo, one_cluster_risk_shao, zishenchai_threshold, zishenwending_subbag, chayi_ratio, j, proportions_new):
    """
    稳定性评估
    """
    if torch.abs(one_cluster_risk_duo - one_cluster_risk_shao) < zishenchai_threshold:
        zishenwending_subbag.append(j)
    else:
        chayi_ratio += proportions_new[j]
    return zishenwending_subbag, chayi_ratio

def assess_enhance_reduction_effect(milnet, zengqiang_result, jiangdi_result, device,
                                    biaozhun_result, j, proportions_new, high_risk_biaozhi, 
                                    low_risk_biaozhi, zengqiangwending_subbag, jiangdiwending_subbag, 
                                    confounding_zhengti, chayi_ratio):
    """
    增强/减弱影响评估
    """
    # for z in range(len(zengqiang_result)):
    
    high_risk_biaozhi.append((milnet(zengqiang_result[j].to(device)) - biaozhun_result)/proportions_new[j])
    low_risk_biaozhi.append((milnet(jiangdi_result[j].to(device)) - biaozhun_result)/proportions_new[j])
    if high_risk_biaozhi[-1] < 0 and low_risk_biaozhi[-1]>0: ##变化趋势相反才是正常的
        zengqiangwending_subbag.append(j)
    elif high_risk_biaozhi[-1] > 0 and low_risk_biaozhi[-1]<0: ##变化趋势相反才是正常的
        jiangdiwending_subbag.append(j)
    else:
        confounding_zhengti.append(j)
        chayi_ratio += proportions_new[j]
    return confounding_zhengti, chayi_ratio

def compute_intersection_difference(zishenwending_subbag, zengqiangwending_subbag, jiangdiwending_subbag):
    """
    计算并集和交集
    """
    result_list = list(set(zishenwending_subbag) & (set(zengqiangwending_subbag) | set(jiangdiwending_subbag)))
    difference_list = list(set(zishenwending_subbag) - set(zengqiangwending_subbag) - set(jiangdiwending_subbag))
    return result_list, difference_list

def store_stable_factors(result_list, dandujulei_duo, high_risk_ratio, low_risk_ratio):
    """
    稳定因素存储
    """
    for idx in result_list:
        if dandujulei_duo[idx] < high_risk_ratio and dandujulei_duo[idx] > low_risk_ratio:
            print(f"Stable cluster: {idx}, Risk: {dandujulei_duo[idx]}")


import numpy as np

def generate_normalized_distribution(*lengths, biaozhi=True, random_dist=False,biaozhi_zhengxiang=False):
    """
    生成基于集合长度的正态分布，并归一化为采样概率分布。
    或者生成随机的概率分布。
    
    参数:
        lengths (tuple of int): 各个集合的长度。
        biaozhi (bool): 标志位，控制是否基于正态分布采样。
        random_dist (bool): 标志位，控制是否随机生成分布。

    返回:
        probabilities (list of float): 采样的概率分布。
    """
    lengths = np.array(lengths)
    means = lengths
    # 处理长度为0的情况
    non_zero_mask = lengths > 0
    std_dev = 1  # 标准正态分布的标准差
    samples = []
    
    if random_dist:
        # 如果启用了随机分布生成
        probabilities = np.random.random(len(lengths))  # 生成随机数
        total_random = np.sum(probabilities)
        probabilities = probabilities / total_random  # 归一化
        probabilities=probabilities.tolist()
    elif biaozhi and np.any(non_zero_mask):
        # 基于非零长度集合的均值生成正态分布
        # samples = np.random.normal(1 / lengths[non_zero_mask], 1)
        # samples = np.clip(samples, 1e-6, None)  # 防止出现负值或过小的值
        
        # total_sample = np.sum(samples)
        # probabilities = np.zeros_like(lengths, dtype=float)
        # probabilities[non_zero_mask] = samples / total_sample
        epsilon = 1e-8
        # 对每个元素取倒数
        # 避免除以零，将零元素替换为一个非常小的数
        lengths = np.where(lengths == 0, np.finfo(float).eps, lengths)
        reciprocal = 1 / lengths
        
        # 进行归一化 (将数组缩放到 0 到 1 之间)
        means = reciprocal / np.sum(reciprocal)

        for mean in means:
            if mean <= epsilon:  # 如果 mean 过小，避免除以 0
                samples.append(0.0)
            else:
                sample = np.random.normal(1/mean, std_dev)
                samples.append(sample)
        # samples = [np.random.normal(1/mean, std_dev) for mean in means]
        # 将采样的值归一化为概率分布
        # total_sample = sum(samples)
        total_sample = sum(samples)
        if total_sample == 0:  # 避免归一化过程中总和为 0
            probabilities = [0 for sample in samples]
        else:
            probabilities = [sample / total_sample for sample in samples]
        # probabilities = [sample / total_sample for sample in samples]
    elif biaozhi_zhengxiang:
        means = means / np.sum(means)
        samples = [np.random.normal(mean, std_dev) for mean in means]
        # 将采样的值归一化为概率分布
        total_sample = sum(samples)
        probabilities = [sample / total_sample for sample in samples]
    else:
        # 平均分配概率
        non_zero_count = np.sum(non_zero_mask)
        if non_zero_count > 0:
            probabilities = np.zeros_like(lengths, dtype=float)
            probabilities[non_zero_mask] = 1 / non_zero_count
        else:
            probabilities = np.zeros_like(lengths, dtype=float)
        probabilities=probabilities.tolist()
    return probabilities


def sample_and_mix_with_normal_distribution(
    gaofengxian_jihe, difengxian_jihe, zhongjian_jihe,
    num_samples, milnet, device,batch_size,biaozhi=True, random_dist=False,biaozhi_zhengxiang=False):
    """
    按照生成的正态分布采样比例进行采样，并计算混合后的伪标签。
    
    参数:
        gaofengxian_jihe (dict): 高风险集合，键为标签，值为张量。
        difengxian_jihe (dict): 低风险集合，键为标签，值为张量。
        zhongjian_jihe (dict): 中间风险集合，键为标签，值为张量。
        num_samples (int): 总采样数量。
        milnet (torch.nn.Module): 用于计算混合结果的模型。
        device (torch.device): 模型所用的设备。

    返回:
        aug_result (torch.Tensor): 模型预测结果。
        weighted_sum (float): 混合后的伪标签加权和。
    """

    def safe_sample(data_jihe, num):
        if len(data_jihe) == 0 or num <= 0:
            return []
        num = min(num, len(data_jihe))  # 防止采样数量超过集合长度
        return random.sample(list(data_jihe.items()), num)

    # 获取各个集合的长度
    lengths = [len(gaofengxian_jihe), len(difengxian_jihe), len(zhongjian_jihe)]

    aug_result_batch = []
    weighted_sum_batch = []
    ## 默认生成的标签rec为1
    rec_batch= []
    # 生成基于正态分布的采样概率分布
    for i in range(batch_size):
        probabilities = generate_normalized_distribution(*lengths,biaozhi=biaozhi, random_dist=random_dist,biaozhi_zhengxiang=biaozhi_zhengxiang)

        # 根据概率分布计算每个集合需要采样的数量
        num_gaofengxian = int(probabilities[0] * num_samples)
        num_difengxian = int(probabilities[1] * num_samples)
        num_zhongjian = num_samples - num_gaofengxian - num_difengxian

        # 从每个集合中采样
        selected_gaofengxian = safe_sample(gaofengxian_jihe, num_gaofengxian)
        selected_difengxian = safe_sample(difengxian_jihe, num_difengxian)
        selected_zhongjian = safe_sample(zhongjian_jihe, num_zhongjian)

        # 拼接选择的张量
        selected_items = selected_gaofengxian + selected_difengxian + selected_zhongjian
        if not selected_items:
            # raise ValueError("选定的集合都为空，无法采样")
            # for i in range(3): ## 直接看谁不是0
            if lengths[0]>0:
                selected_items = safe_sample(gaofengxian_jihe, 10)
            if lengths[1]>0:
                selected_items = safe_sample(difengxian_jihe, 10)
            if lengths[2]>0:
                selected_items = safe_sample(zhongjian_jihe, 10)
        selected_tensors = [value for key, value in selected_items]
        concatenated_tensor = torch.cat(selected_tensors, dim=0)

        # 计算每个张量长度所占的比例
        total_length = sum([tensor.size(0) for tensor in selected_tensors])
        weighted_sum = sum(
            (key.item() * tensor.size(0) / total_length) 
            for key, tensor in selected_items
        )

        # 将拼接后的张量传入模型
        # aug_result = milnet(concatenated_tensor.to(device))
        aug_result_batch.append(concatenated_tensor.to(device))
        weighted_sum_batch.append(torch.tensor(weighted_sum).to(device))
        rec_batch.append(torch.tensor(1).to(device))
    return aug_result_batch, weighted_sum_batch,rec_batch


def causal_aug(data_collector, rec_collector, PFS_collector, idx_collector,  
               milnet, clu_num, device, batch_size, mode,
               Stable_threshold=0.025, selected_ratio_causal=0.5,
               zishenchai_threshold=0.02, high_risk_ratio=0.07, low_risk_ratio=0.27, biaozhi=True, random_dist=False, biaozhi_zhengxiang=False):
    """
    This function performs causal data augmentation by evaluating the stability and risk of patients,
    and then generating new samples based on the risk groups (high, medium, low). It aims to create a
    balanced augmented dataset to improve the robustness of the model.
    
    data_collector: list
    A list containing the input data for each patient (e.g., features or covariates).
    [{'x':tensor(1,n,1024),'cluster_level1_10':tensor(1,n),'cluster_level2_10':tensor(1,n)},...] Length of batch_size
    rec_collector: list
        A list containing the recurrence information for each patient (e.g., whether the patient experienced recurrence or not).
    [tensor(0., device='cuda:7'),...,tensor(1., device='cuda:7')] Length of batch_size
    PFS_collector: list
        A list containing the progression-free survival (PFS) data for each patient (i.e., the time until disease progression or death).
    [tensor(0.3645., device='cuda:7'),...,tensor(0.0012, device='cuda:7')] Length of batch_size
    idx_collector: list
        A list containing the index or unique identifier for each patient, used to track their data across various stages of processing.
    [('TCGA-A2-A04W',)('TCGA-A2-A0EN',),...] Length of batch_size
    milnet: torch.nn.Module
        A pre-trained model (e.g., a neural network) that is used to make predictions on the data (likely a model for survival analysis or risk prediction).

    clu_num: int
        The number of clusters or groups to be used in the causal augmentation process, typically related to clustering patients based on similar features or behaviors.

    device: torch.device
        The device (CPU or GPU) on which computations will be performed (e.g., 'cuda' for GPU or 'cpu' for CPU).

    batch_size: int
        The number of samples to be processed in each batch during model evaluation or data augmentation.

    mode: str
        Specifies the mode in which the function operates. This could determine whether the process is for training, validation, or testing.
    TGCA-BRCA setup
    Stable_threshold: float, optional (default=0.025)
        A threshold value for determining the stability of a patient's data (e.g., variance in predictions). If the stability is below this threshold, the patient is considered stable.

    selected_ratio_causal: float, optional (default=0.5)
        A ratio indicating how much of the causal augmentation should be selected for the current iteration. This parameter controls how much new causal data is generated for the model.

    zishenchai_threshold: float, optional (default=0.02)
        A threshold for evaluating the difference between sub-bags in the stability assessment (likely comparing different clusters).

    high_risk_ratio: float, optional (default=0.07)
        The ratio that defines high-risk groups based on the predictions. It helps categorize patients into high-risk groups based on their predicted outcomes.

    low_risk_ratio: float, optional (default=0.27)
        The ratio that defines low-risk groups based on the predictions. Similar to `high_risk_ratio`, it categorizes patients into low-risk groups.

    biaozhi: bool, optional (default=True)
        A flag indicating whether to apply a standardization or labeling process for the augmented data, likely used for consistency in data labeling.

    random_dist: bool, optional (default=False)
        A flag indicating whether to apply a random distribution when mixing or augmenting the data. If `True`, data samples are mixed randomly.

    biaozhi_zhengxiang: bool, optional (default=False)
        A flag indicating whether to apply a specific standardization or correction process (likely related to the data being processed in a certain way).

    """
    
    milnet.eval()  # Set the model to evaluation mode
    i_batch = 0  # Initialize batch index
    res = {}  # Dictionary to store results for each patient
    wending_zhongjian_jihe = {}  # Set for intermediate stability factors
    wending_gaofengxian_jihe = {}  # Set for high-risk stability factors
    wending_difengxian_jihe = {}  # Set for low-risk stability factors
    
    # Generate random ratios for augmentation
    ratios = [round(random.uniform(0.5, 1.0), 2) for _ in range(100)]
    
    # Create a pseudo-bag for data clustering with the ProtoDiv method
    pseudo_bag = PseudoBag_tlz(n=clu_num, l=clu_num, clustering_method='ProtoDiv', 
                               device=device, PF=[6, 5, 4, 3, 2, 1, 0.5, 0.1], 
                               noise_level=0.1, n_clusters=clu_num, sample_proportion=0.8)
    
    with torch.no_grad():  # Disable gradient tracking for evaluation
        # Loop through each batch of data
        for data, rec, PFS, graph_path in zip(data_collector, rec_collector, PFS_collector, idx_collector):
            res[graph_path[0]] = {}  # Initialize dictionary for patient results
            
            # Prepare input data and move to the device (GPU/CPU)
            data_x = data['x'].to(device)
            data_rec = rec
            data_PFS = PFS
            i_batch += 1
            graph = data
            data_x = data_x.to(device)
            data_rec = data_rec.to(device).squeeze()
            data_PFS = data_PFS.to(device).squeeze()
            
            # Extract clustering levels from the data
            data_cluster_level1 = data['cluster_level1_10']
            data_cluster_level2 = data['cluster_level2_10']
            graph[f'cluster_level3_10'] = -1 * torch.ones(len(graph['x'].squeeze())).unsqueeze(dim=0).cpu().long()

            # Generate pseudo-bag predictions (simulating clusters and their labels)
            y_hat = generate_pseudo_bag(data_x, data_cluster_level1, pseudo_bag, milnet, ratios, mode, cishu=10)
            
            # Calculate the mean and variance of the pseudo-bag predictions
            mean = sum(y_hat) / len(y_hat)
            variance = sum((x - mean) ** 2 for x in y_hat) / (len(y_hat) - 1)
            
            # Evaluate stability of the patient's cluster (if stable, perform causal augmentation)
            Stable_patient, mean, variance = evaluate_stability(y_hat, Stable_threshold)
            
            if Stable_patient:  # If the patient is stable, proceed with causal augmentation
                biaozhun_bag, zengqiang_result, jiangdi_result, dandujulei_duo, dandujulei_shao, no_julei, proportions_new = \
                    pseudo_bag.Control_a_category_increases_decreases(
                        graph, sample_proportion=selected_ratio_causal)
                
                biaozhun_result = milnet(biaozhun_bag.to(device))
                
                # Lists to store different risk factors
                high_risk, low_risk, confounding = [], [], []
                high_risk_biaozhi, low_risk_biaozhi = [], []
                confounding_zhengti = []
                one_cluster_risk_duo_result, one_cluster_risk_shao_result = [], []
                
                zishenwending_subbag, zengqiangwending_subbag, jiangdiwending_subbag = [], [], []
                
                sum_test, high_ratio, low_ratio, confounding_ratio, chayi_ratio = 0, 0, 0, 0, 0
                
                # Loop through the enhanced and reduced clusters to assess risk and stability
                for j in range(len(zengqiang_result)):
                    if len(dandujulei_shao[j]) > 0:  # Ensure there are enough samples for each cluster
                        # Evaluate risk for the enhanced and reduced sub-clusters
                        one_cluster_risk_duo = milnet(dandujulei_duo[j].to(device))
                        one_cluster_risk_shao = milnet(dandujulei_shao[j].to(device))
                        one_cluster_risk_duo_result.append(one_cluster_risk_duo)
                        one_cluster_risk_shao_result.append(one_cluster_risk_shao)
                        
                        # Assess risk based on the predictions
                        high_ratio, low_ratio, confounding_ratio = assess_risk(
                            one_cluster_risk_duo, one_cluster_risk_shao, 
                            high_risk_ratio, low_risk_ratio, 
                            high_ratio, low_ratio, confounding_ratio, 
                            high_risk, low_risk, confounding, j, proportions_new
                        )
                        
                        # Assess stability of the clusters
                        zishenwending_subbag, chayi_ratio = assess_stability(
                            one_cluster_risk_duo, one_cluster_risk_shao, 
                            zishenchai_threshold, zishenwending_subbag, 
                            chayi_ratio, j, proportions_new
                        )
                        
                        # Assess the enhancement or reduction effects
                        confounding_zhengti, chayi_ratio = assess_enhance_reduction_effect(
                            milnet, zengqiang_result, jiangdi_result, device,
                            biaozhun_result, j, proportions_new, 
                            high_risk_biaozhi, low_risk_biaozhi, 
                            zengqiangwending_subbag, jiangdiwending_subbag, 
                            confounding_zhengti, chayi_ratio
                        )

                    else:
                        continue
                
                # Compute intersection and difference of stable factors across different risk groups
                result_list, difference_list = compute_intersection_difference(
                    zishenwending_subbag, zengqiangwending_subbag, jiangdiwending_subbag
                )
                
                # Store stable factors in the appropriate risk group sets
                if len(result_list) > 0:
                    for k in range(len(result_list)):
                        if one_cluster_risk_duo_result[result_list[k]] < high_risk_ratio:
                            wending_gaofengxian_jihe[one_cluster_risk_duo_result[result_list[k]]] = dandujulei_duo[result_list[k]].to(device)
                        elif one_cluster_risk_duo_result[result_list[k]] > low_risk_ratio:
                            wending_difengxian_jihe[one_cluster_risk_duo_result[result_list[k]]] = dandujulei_duo[result_list[k]].to(device)
                        else:
                            wending_zhongjian_jihe[one_cluster_risk_duo_result[result_list[k]]] = dandujulei_duo[result_list[k]].to(device)
                
                # Continue further exploration if needed (currently disabled)
                if False:
                    wending_gaofengxian_jihe, wending_difengxian_jihe, wending_zhongjian_jihe = further_exploration(
                        milnet, device, difference_list, data_x, data_cluster_level1, data_cluster_level2, pseudo_bag, 
                        selected_ratio_causal, wending_gaofengxian_jihe, wending_difengxian_jihe, wending_zhongjian_jihe,
                        high_risk_ratio, low_risk_ratio, zishenchai_threshold
                    )
                
            else:
                continue  # If the patient is not stable, skip to the next patient
            
        # 1. Randomly sample and mix elements (e.g., select 2 elements)
        if len({**wending_gaofengxian_jihe, **wending_difengxian_jihe, **wending_zhongjian_jihe}) > 0:
            # Perform data augmentation by sampling from the different risk groups
            aug_result, weighted_sum, rec_batch = sample_and_mix_with_normal_distribution(
                wending_gaofengxian_jihe, wending_difengxian_jihe, wending_zhongjian_jihe,
                num_samples=clu_num, milnet=milnet, device=device, batch_size=batch_size, biaozhi=biaozhi,
                random_dist=random_dist, biaozhi_zhengxiang=biaozhi_zhengxiang,
            )
        else:
            aug_result, weighted_sum, rec_batch = [], [], []  # No samples if no stable factors are found
        
        # Final augmentation is complete;
    torch.cuda.empty_cache()

    return aug_result, weighted_sum ,rec_batch
