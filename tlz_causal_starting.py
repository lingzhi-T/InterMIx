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

"The rest of the code will be made public after the acceptance of the paper"
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
