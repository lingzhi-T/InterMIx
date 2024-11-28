# Intervention-based-Mixup-Augmentation-for-Multiple-Instance-Learning-in-Whole-Slide-Image-Time-to-Event Analysis
Code for Intervention-based Mixup Augmentation for Multiple Instance Learning in Whole-Slide Image Survival Analysis

## The feature extraction process is referenced to [CLAM](https://github.com/mahmoodlab/CLAM) (Take TCGA-BRCA for example)
```
python create_patches_fp.py     --source /home/daoyjz03/CLAM/svs_file/bingli_20240729/brca    --save_dir /home/daoyjz03/CLAM/svs_file/bingli_20240729/brca_tiles-l1-s256_new     --patch_level 1 --patch_size 256 --seg --patch --stitch

CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py \
    --data_h5_dir /home/daoyjz03/CLAM/svs_file/bingli_20240729/brca_tiles-l1-s256_new \
    --data_slide_dir /home/daoyjz03/CLAM/svs_file/bingli_20240729/brca \
    --csv_path /home/daoyjz03/CLAM/svs_file/bingli_20240729/brca_tiles-l1-s256_new/process_list_autogen.csv \
    --feat_dir /home/daoyjz03/CLAM/svs_file/bingli_20240729/brca_processed/feat-l1-unvi-B-new \
    --batch_size 512 --slide_ext .svs
```
## The data were divided and preprocessed with reference to [AdvMIL](https://github.com/liupei101/AdvMIL)
```
from load_dataset import get_dataloader_tlz_surv
mode = 'TransMIL'  # 'DSMIL' 'ABMIL' 'CLAM'
dataloader = get_dataloader_tlz_surv(index=index,
                                    path_split='data_split/tcga_brca-fold',  ## Tables are from ADVMIL
                                    table_path = 'data_split/tcga_brca_path_full.csv',  ## Tables are from ADVMIL
                                    train_dir="/home/tlz/CLAM/svs_file/bingli_20240729/brca_processed/pt_files_cluster_10", 
                                    mode=mode)  ## 'mode' is because the input of remix and other inconsistent, so take a flag to determine, normal use do not care about

```
## Introduction to the process of use
```
# It is recommended to use two variables to control the frequency of epoch, and the frequency of batch
epoch_pingci = 4
batch_pingci = 2
n_batch = 0
...
for ei in range(0,epoch):
    milnet.train()
    milnet.to(device)
    
    ...
    for data,bag_label,graph_path in dataloader["train"]:
      n_batch += 1
      ...
      if ei>=warmup_epoch and ei%epoch_pingci==0 and n_batch%batch_pingci==1: ## Controlling the frequency of data enhancement
          ## For a detailed explanation of the parameters see tlz_causal_starting.py
          x_aug, PFS_collector ,rec_collector = causal_aug(data_collector, rec_collector, PFS_collector,idx_collector,
                                                           milnet,mode=mode,device=device,batch_size=bp_every_batch,
                                                           clu_num=cnum,high_risk_ratio =0.07,low_risk_ratio=0.27)  
          ## It is recommended to adjust high_risk_ratio , low_risk_ratio according to the data set,
          ##  BRCA we used high_risk_ratio =0.07,low_risk_ratio=0.27, LGG :high_risk_ratio =0.03 low_risk_ratio=0.22
          milnet.train()
          if len(x_aug)==n_sample:
              for i in range(n_sample):              
                  logit_bag = milnet(x_aug[i].squeeze())
                  y_hat.append(logit_bag)
      else:  ## No data augmentation
          y_hat = []
          if mode =='cluster':
              for i in range(n_sample):              
                  logit_bag = milnet(x_collector[i].squeeze(),cluster_level1_10_collector[i])
                  y_hat.append(logit_bag)
          else:
              for i in range(n_sample):              
                  logit_bag = milnet(x_collector[i].squeeze())
                  y_hat.append(logit_bag)
    ...
    ei+=1
    
```


