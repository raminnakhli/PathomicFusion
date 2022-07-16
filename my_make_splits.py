### data_loaders.py
import argparse
import os
import pickle

import numpy as np
import pandas as pd
# Env
import torch
from PIL import Image
from sklearn import preprocessing
from torchvision import transforms

### Initializes parser and data
"""
all_st
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st # for training Surv Path, Surv Graph, and testing Surv Graph
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st # for training Grad Path, Grad Graph, and testing Surv_graph
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st # for training Surv Omic, Surv Graphomic
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st # for training Grad Omic, Grad Graphomic

all_st_patches_512 (no VGG)
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st_patches_512 # for testing Surv Path
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st_patches_512 # for testing Grad Path

all_st_patches_512 (use VGG)
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15 --gpu_ids 0 # for Surv Pathgraph
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --act_type LSM --label_dim 3 --gpu_ids 1 # for Grad Pathgraph
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15 --gpu_ids 2 # for Surv Pathomic, Pathgraphomic
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --act_type LSM --label_dim 3 --gpu_ids 3 # for Grad Pathomic, Pathgraphomic


python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --make_all_train 1

python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15 --use_rnaseq 1 --gpu_ids 2
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --use_rnaseq 1 --act_type LSM --label_dim 3 --gpu_ids 3


python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --gpu_ids 0
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --use_rnaseq 1 --gpu_ids 0

python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --act_type LSM --label_dim 3 --gpu_ids 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --use_rnaseq 1 --act_type LSM --label_dim 3 --gpu_ids 1

python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --gpu_ids 2

python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --act_type LSM --label_dim 3 --gpu_ids 3




"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data/TCGA_GBMLGG/', help="datasets")
    parser.add_argument('--roi_dir', type=str, default='all_st')
    parser.add_argument('--graph_feat_type', type=str, default='cpc', help="graph features to use")
    parser.add_argument('--ignore_missing_moltype', type=int, default=0,
                        help="Ignore data points with missing molecular subtype")
    parser.add_argument('--ignore_missing_histype', type=int, default=0,
                        help="Ignore data points with missign histology subtype")
    parser.add_argument('--make_all_train', type=int, default=0)
    parser.add_argument('--use_vgg_features', type=int, default=0)
    parser.add_argument('--use_rnaseq', type=int, default=0)

    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/TCGA_GBMLGG/',
                        help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='surv_15_rnaseq',
                        help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='path', help='mode')
    parser.add_argument('--model_name', type=str, default='path', help='mode')
    parser.add_argument('--task', type=str, default='surv', help='surv | grad')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--batch_size', type=int, default=32, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--init_type', type=str, default='none',
                        help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.25, type=float,
                        help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')

    opt = parser.parse_known_args()[0]
    return opt


opt = parse_args()

patient_folds = {
    '0': {
        'train': [326.0, 327.0, 328.0, 329.0, 330.0, 331.0, 332.0, 334.0, 335.0, 336.0, 342.0, 343.0, 344.0, 345.0,
                  348.0, 349.0, 350.0, 351.0, 352.0, 353.0, 355.0, 366.0, 369.0, 371.0, 372.0, 374.0, 376.0, 378.0,
                  379.0, 380.0, 382.0, 383.0, 384.0, 385.0, 388.0, 389.0, 390.0, 392.0, 393.0, 394.0, 396.0, 402.0,
                  404.0, 406.0, 407.0, 413.0, 414.0, 416.0, 417.0, 419.0, 421.0, 422.0, 423.0, 425.0, 426.0, 427.0,
                  428.0, 429.0, 431.0, 432.0, 434.0, 435.0, 436.0, 437.0, 439.0, 440.0, 441.0, 445.0, 446.0, 447.0,
                  448.0, 450.0, 453.0, 454.0, 458.0, 459.0, 461.0, 462.0, 464.0, 468.0, 479.0, 480.0, 481.0, 482.0,
                  483.0, 485.0, 487.0, 489.0, 493.0, 494.0, 495.0, 496.0, 498.0, 499.0, 501.0, 502.0, 503.0, 504.0,
                  505.0, 506.0, 507.0, 515.0, 517.0, 518.0, 521.0, 523.0, 525.0, 526.0, 528.0, 530.0, 533.0, 534.0,
                  535.0, 536.0, 539.0, 540.0, 544.0, 545.0, 546.0, 553.0, 554.0, 555.0, 556.0, 562.0, 571.0, 573.0,
                  574.0, 579.0, 582.0, 583.0, 584.0, 585.0, 594.0, 601.0, 604.0, 609.0, 614.0, 617.0, 619.0, 620.0,
                  621.0, 626.0, 627.0, 629.0, 635.0, 636.0, 641.0, 650.0, 652.0, 654.0, 656.0, 657.0, 658.0, 659.0,
                  661.0, 662.0, 665.0, 666.0, 667.0, 670.0, 672.0, 678.0, 681.0, 683.0, 685.0, 686.0, 687.0, 689.0,
                  695.0, 697.0, 698.0, 701.0, 716.0, 720.0, 724.0, 725.0, 727.0, 728.0, 729.0, 730.0, 731.0, 734.0,
                  738.0, 739.0, 744.0, 746.0, 749.0, 752.0, 753.0, 754.0, 759.0, 762.0, 765.0, 777.0, 779.0, 782.0,
                  783.0, 784.0, 785.0, 786.0, 790.0, 795.0, 796.0, 798.0, 799.0, 800.0, 801.0, 804.0, 806.0, 807.0],
        'test': [189.0, 190.0, 191.0, 40.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 201.0, 202.0, 203.0, 205.0, 206.0,
                 207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 2.0, 217.0, 46.0, 218.0, 219.0, 221.0, 223.0, 232.0, 236.0,
                 239.0, 48.0, 241.0, 242.0, 255.0, 260.0, 268.0, 269.0, 273.0, 278.0, 280.0, 51.0, 281.0, 282.0, 283.0,
                 284.0, 285.0, 291.0, 292.0, 293.0, 53.0, 294.0, 295.0, 296.0, 297.0, 300.0, 305.0, 306.0, 307.0, 308.0,
                 309.0, 55.0, 319.0, 321.0, 322.0, 323.0, 324.0, 325.0, 56.0, 62.0, 65.0, 67.0, 8.0, 68.0, 69.0, 70.0,
                 73.0, 76.0, 77.0, 79.0, 80.0, 89.0, 90.0, 91.0, 94.0, 95.0, 100.0, 12.0, 101.0, 15.0, 18.0, 23.0, 24.0,
                 171.0, 172.0, 173.0, 175.0, 176.0, 178.0, 179.0, 34.0, 180.0, 182.0, 184.0, 186.0, 188.0],
    },
    '1': {
        'train': [189.0, 190.0, 191.0, 40.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 201.0, 202.0, 203.0, 205.0,
                  206.0, 207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 2.0, 217.0, 46.0, 218.0, 219.0, 221.0, 223.0, 232.0,
                  236.0, 239.0, 48.0, 241.0, 242.0, 255.0, 260.0, 268.0, 269.0, 273.0, 278.0, 280.0, 51.0, 281.0, 282.0,
                  283.0, 284.0, 285.0, 291.0, 292.0, 293.0, 53.0, 294.0, 295.0, 296.0, 297.0, 300.0, 305.0, 306.0,
                  307.0, 308.0, 309.0, 55.0, 319.0, 321.0, 322.0, 323.0, 324.0, 325.0, 56.0, 62.0, 65.0, 67.0, 8.0,
                  68.0, 69.0, 70.0, 73.0, 76.0, 77.0, 79.0, 80.0, 523.0, 525.0, 526.0, 528.0, 530.0, 533.0, 534.0,
                  535.0, 536.0, 539.0, 540.0, 544.0, 545.0, 546.0, 553.0, 554.0, 555.0, 556.0, 562.0, 571.0, 573.0,
                  574.0, 579.0, 582.0, 583.0, 584.0, 585.0, 594.0, 601.0, 604.0, 609.0, 614.0, 617.0, 619.0, 620.0,
                  621.0, 626.0, 627.0, 629.0, 89.0, 635.0, 636.0, 641.0, 650.0, 652.0, 654.0, 656.0, 657.0, 658.0,
                  659.0, 90.0, 661.0, 662.0, 665.0, 666.0, 667.0, 670.0, 672.0, 678.0, 681.0, 91.0, 683.0, 685.0, 686.0,
                  687.0, 689.0, 695.0, 697.0, 94.0, 698.0, 701.0, 716.0, 720.0, 724.0, 725.0, 727.0, 728.0, 95.0, 729.0,
                  730.0, 731.0, 734.0, 738.0, 739.0, 744.0, 100.0, 746.0, 749.0, 12.0, 752.0, 753.0, 754.0, 759.0,
                  762.0, 765.0, 101.0, 777.0, 779.0, 782.0, 783.0, 784.0, 785.0, 786.0, 790.0, 795.0, 796.0, 798.0,
                  799.0, 800.0, 801.0, 804.0, 806.0, 807.0, 15.0, 18.0, 23.0, 24.0, 171.0, 172.0, 173.0, 175.0, 176.0,
                  178.0, 179.0, 34.0, 180.0, 182.0, 184.0, 186.0, 188.0],
        'test': [326.0, 327.0, 328.0, 329.0, 330.0, 331.0, 332.0, 334.0, 335.0, 336.0, 342.0, 343.0, 344.0, 345.0,
                 348.0, 349.0, 350.0, 351.0, 352.0, 353.0, 355.0, 366.0, 369.0, 371.0, 372.0, 374.0, 376.0, 378.0,
                 379.0, 380.0, 382.0, 383.0, 384.0, 385.0, 388.0, 389.0, 390.0, 392.0, 393.0, 394.0, 396.0, 402.0,
                 404.0, 406.0, 407.0, 413.0, 414.0, 416.0, 417.0, 419.0, 421.0, 422.0, 423.0, 425.0, 426.0, 427.0,
                 428.0, 429.0, 431.0, 432.0, 434.0, 435.0, 436.0, 437.0, 439.0, 440.0, 441.0, 445.0, 446.0, 447.0,
                 448.0, 450.0, 453.0, 454.0, 458.0, 459.0, 461.0, 462.0, 464.0, 468.0, 479.0, 480.0, 481.0, 482.0,
                 483.0, 485.0, 487.0, 489.0, 493.0, 494.0, 495.0, 496.0, 498.0, 499.0, 501.0, 502.0, 503.0, 504.0,
                 505.0, 506.0, 507.0, 515.0, 517.0, 518.0, 521.0]
    },
    '2': {
        'train': [189.0, 190.0, 191.0, 40.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 201.0, 202.0, 203.0, 205.0,
                  206.0, 207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 2.0, 217.0, 46.0, 218.0, 219.0, 221.0, 223.0, 232.0,
                  236.0, 239.0, 48.0, 241.0, 242.0, 255.0, 260.0, 268.0, 269.0, 273.0, 278.0, 280.0, 51.0, 281.0, 282.0,
                  283.0, 284.0, 285.0, 291.0, 292.0, 293.0, 53.0, 294.0, 295.0, 296.0, 297.0, 300.0, 305.0, 306.0,
                  307.0, 308.0, 309.0, 55.0, 319.0, 321.0, 322.0, 323.0, 324.0, 325.0, 326.0, 327.0, 56.0, 328.0, 329.0,
                  330.0, 331.0, 332.0, 334.0, 335.0, 336.0, 342.0, 62.0, 343.0, 344.0, 345.0, 348.0, 349.0, 350.0,
                  351.0, 352.0, 353.0, 355.0, 65.0, 366.0, 369.0, 371.0, 372.0, 374.0, 376.0, 67.0, 378.0, 379.0, 380.0,
                  382.0, 383.0, 8.0, 384.0, 385.0, 388.0, 389.0, 390.0, 68.0, 392.0, 393.0, 394.0, 396.0, 402.0, 404.0,
                  69.0, 406.0, 407.0, 413.0, 414.0, 416.0, 417.0, 70.0, 419.0, 421.0, 422.0, 423.0, 425.0, 426.0, 427.0,
                  428.0, 429.0, 431.0, 432.0, 434.0, 435.0, 436.0, 437.0, 439.0, 440.0, 441.0, 73.0, 445.0, 446.0,
                  447.0, 448.0, 450.0, 453.0, 454.0, 458.0, 459.0, 461.0, 462.0, 464.0, 468.0, 76.0, 479.0, 480.0,
                  481.0, 482.0, 483.0, 485.0, 77.0, 487.0, 489.0, 493.0, 494.0, 495.0, 496.0, 498.0, 499.0, 79.0, 501.0,
                  502.0, 503.0, 504.0, 505.0, 506.0, 507.0, 515.0, 80.0, 517.0, 518.0, 521.0, 89.0, 90.0, 91.0, 94.0,
                  95.0, 100.0, 12.0, 101.0, 15.0, 18.0, 23.0, 24.0, 171.0, 172.0, 173.0, 175.0, 176.0, 178.0, 179.0,
                  34.0, 180.0, 182.0, 184.0, 186.0, 188.0],
        'test': [523.0, 525.0, 526.0, 528.0, 530.0, 533.0, 534.0, 535.0, 536.0, 539.0, 540.0, 544.0, 545.0, 546.0,
                 553.0, 554.0, 555.0, 556.0, 562.0, 571.0, 573.0, 574.0, 579.0, 582.0, 583.0, 584.0, 585.0, 594.0,
                 601.0, 604.0, 609.0, 614.0, 617.0, 619.0, 620.0, 621.0, 626.0, 627.0, 629.0, 635.0, 636.0, 641.0,
                 650.0, 652.0, 654.0, 656.0, 657.0, 658.0, 659.0, 661.0, 662.0, 665.0, 666.0, 667.0, 670.0, 672.0,
                 678.0, 681.0, 683.0, 685.0, 686.0, 687.0, 689.0, 695.0, 697.0, 698.0, 701.0, 716.0, 720.0, 724.0,
                 725.0, 727.0, 728.0, 729.0, 730.0, 731.0, 734.0, 738.0, 739.0, 744.0, 746.0, 749.0, 752.0, 753.0,
                 754.0, 759.0, 762.0, 765.0, 777.0, 779.0, 782.0, 783.0, 784.0, 785.0, 786.0, 790.0, 795.0, 796.0,
                 798.0, 799.0, 800.0, 801.0, 804.0, 806.0, 807.0]
    },
}

pnas_splits = pd.DataFrame({
    'TCGA ID': np.unique(patient_folds['0']['train'] + patient_folds['0']['test']).astype('int'),
})
pnas_splits['0'], pnas_splits['1'], pnas_splits['2'] = 'Train', 'Train', 'Train'

for k, v in patient_folds.items():
    pnas_splits.loc[pnas_splits['TCGA ID'].isin(patient_folds[k]['test']), k] = 'Test'

study_to_core_id = pd.read_csv('core_id.csv')
outcome = pd.read_csv('outcome.csv')

cv_splits = {}
graph_path = ''


def core_id_to_study_id(st):
    return int(study_to_core_id.loc[study_to_core_id.core_id == st].study_id.values[0])


def list_files(dir, ext):
    return [os.path.join(dir, f) for f in os.listdir(dir) if dir.endswith(ext)]


for k, v in patient_folds.items():
    train_list = []
    train_pt_name = []
    train_time_list = []
    train_censor_list = []
    test_list = []
    test_pt_name = []
    test_time_list = []
    test_censor_list = []
    for f in list_files(graph_path, '.pt'):
        filename = os.path.split(os.path.splitext(f)[0])[1]
        study_id = core_id_to_study_id(filename.split('_')[0])
        censor = outcome.loc[outcome.study_id == study_id].status.values[0]
        if censor == 'partial':
            continue
        elif censor == 'censor':
            censor = 0
        elif censor == 'event':
            censor = 1
        time = float(outcome.loc[outcome.study_id == study_id].time.values[0]) * 12
        if study_id in v['train']:
            train_list.append(f)
            train_pt_name.append(str(study_id))
            train_time_list.append(time)
            train_censor_list.append(censor)
        if study_id in v['test']:
            test_list.append(f)
            test_pt_name.append(str(study_id))
            test_time_list.append(time)
            test_censor_list.append(censor)
    cv_splits[int(k)] = {
        'train': {
            'x_patname': train_pt_name,
            'x_path': train_pt_name,
            'x_grph': train_list,
            'x_omic': [''] * len(train_pt_name),
            'e': train_censor_list,
            't': train_time_list,

        }, 'test': {
            'x_patname': test_pt_name,
            'x_path': test_pt_name,
            'x_grph': test_list,
            'x_omic': [''] * len(test_pt_name),
            'e': test_censor_list,
            't': test_time_list,
        }
    }

data_dict = {}
data_dict['cv_splits'] = cv_splits
pickle.dump(data_dict, open('my_split.pkl', 'wb'))
