"""
Experiment configuration file
Extended from config file from original PANet Repository
"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds  #
from platform import node
from datetime import datetime

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('mySSL')
ex.captured_out_filter = apply_backspaces_and_linefeeds

# source_folders = ['/home/wu/source_code/Self-supervised-Fewshot-Medical-Image-Segmentation-master', './dataloaders', './models', './util']
source_folders = ['.', './dataloaders', './models', './util']
# print("#######")
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    seed = 1234
    gpu_id = 0
    mode = 'train' # for now only allows 'train' 
    num_workers = 4 # 0 for debugging. 
    # dataset='CHAOST2_Superpix'
    dataset="SABS_Superpix"
    dataset_1 = 'SABS_Superpix_1' # i.e. abdominal MRI
    dataset_2 = 'SABS_Superpix_2'
    dataset_3 = 'SABS_Superpix_3'  # i.e. abdominal MRI
    dataset_4 = 'SABS_Superpix_4'

    # dataset = ''
    use_coco_init = True # initialize backbone with MS_COCO initialization. Anyway coco does not contain medical images
    rounds = 10  # comunication rounds
    ### Training
    n_steps = 100100
    batch_size = 1
    lr_milestones = [ (ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 100
    save_snapshot_every = 25000
    max_iters_per_load = 1000 # epoch size, interval for reloading the dataset
    scan_per_load = -1 # numbers of 3d scans per load for saving memory. If -1, load the entire dataset to the memory
    which_aug = 'sabs_aug' # standard data augmentation with intensity and geometric transforms
    input_size = (256, 256)
    min_fg_data='1' # when training with manual annotations, indicating number of foreground pixels in a single class single slice. This empirically stablizes the training process
    label_sets = 0 # which group of labels taking as training (the rest are for testing)
    exclude_cls_list = [2, 3] # testing classes to be excluded in training. Set to [] if testing under setting 1
    usealign = True # see vanilla PANet
    use_wce = True

    ### Validation
    z_margin = 0 
    eval_fold = 0 # which fold for 5 fold cross validation
    support_idx=[-1] # indicating which scan is used as support in testing. 
    val_wsize=2 # L_H, L_W in testing
    n_sup_part = 3 # number of chuncks in testing

    # Network
    modelname = 'dlfcn_res101' # resnet 101 backbone from torchvision fcn-deeplab
    clsname = None # 
    reload_model_path = None # path for reloading a trained model (overrides ms-coco initialization)
    # reload_model_path = "/home/wu/source_code/fed_fewshot/glob_weights/w_global.pth"
    # reload_model_path ="/home/wu/source_code/fed_fewshot/transfer_w/transfer_weights.pth"  # use transfered weights
    proto_grid_size = 8 # L_H, L_W = (32, 32) / 8 = (4, 4)  in training
    feature_hw = [32, 32] # feature map size, should couple this with backbone in future

    # SSL
    superpix_scale = 'MIDDLE' #MIDDLE/ LARGE

    model = {
        'align': usealign,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'cls_name': clsname,
        'proto_grid_size' : proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path
    }

    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': 1,
        'npart': n_sup_part 
    }

    optim_type = 'sgd'
    optim = {
        # 'lr': 1e-6,
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }

    exp_prefix = ''

    exp_str = '_'.join(
        [exp_prefix]
        + [dataset,]
        + [f'sets_{label_sets}_{task["n_shots"]}shot'])
    # exp_str = '_'.join(
    #     [exp_prefix]
    #     + [dataset_1, ]
    #     + [f'sets_{label_sets}_{task["n_shots"]}shot'])

    path = {
        'log_dir': '/home/b311/data1/wjp/fed_fewshot_proto/runs',
        'SABS':{'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/SABS/sabs_CT_normalized"
            },
        'C0':{'data_dir': "feed your dataset path here"
            },
        'CHAOST2_1':{'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/CHAOST2/chaos_MR_T2_normalized_1"
            },
        'SABS_Superpix':{'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/SABS/sabs_CT_normalized"},
        'C0_Superpix':{'data_dir': "feed your dataset path here"},
        # 'CHAOST2_Superpix_1':{'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/CHAOST2/chaos_MR_T2_normalized_1"},
        # 'CHAOST2_2':{'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/CHAOST2/chaos_MR_T2_normalized_2"},
        # 'CHAOST2_Superpix_2': {'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/CHAOST2/chaos_MR_T2_normalized_2"},
        # 'CHAOST2_3': {'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/CHAOST2/chaos_MR_T2_normalized_3"},
        # 'CHAOST2_Superpix_3': {'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/CHAOST2/chaos_MR_T2_normalized_3"},
        # 'CHAOST2_4': {'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/CHAOST2/chaos_MR_T2_normalized_4"},
        # 'CHAOST2_Superpix_4': {'data_dir': "/home/wu/source_code/fed_fewshot_proto_demo/data/CHAOST2/chaos_MR_T2_normalized_4"},
        'SABS_1': {'data_dir': "/date/wjp/Fed_fewshot_new/data/SABS/tmp_normalized_Spleen"},
        'SABS_Superpix_1': {'data_dir': "/date/wjp/Fed_fewshot_new/data/SABS/tmp_normalized_Spleen"},
        'SABS_2': {'data_dir': "/date/wjp/Fed_fewshot_new/data/SABS/tmp_normalized_RK"},
        'SABS_Superpix_2': {'data_dir': "/date/wjp/Fed_fewshot_new/data/SABS/tmp_normalized_RK"},
        'SABS_3': {'data_dir': "/date/wjp/Fed_fewshot_new/data/SABS/tmp_normalized_LK"},
        'SABS_Superpix_3': {'data_dir': "/date/wjp/Fed_fewshot_new/data/SABS/tmp_normalized_LK"},
        'SABS_4': {'data_dir': "/date/wjp/Fed_fewshot_new/data/SABS/tmp_normalized_Liver"},
        'SABS_Superpix_4': {'data_dir': "/date/wjp/Fed_fewshot_new/data/SABS/tmp_normalized_Liver"},
        # 'SABS_1': {'data_dir': "/date/wjp/fed_fewshot_proto/data/SABS/sabs_CT_normalized_1"},
        # 'SABS_Superpix_1': {'data_dir': "/date/wjp/fed_fewshot_proto/data/SABS/sabs_CT_normalized_1"},
        # 'SABS_2': {'data_dir': "/date/wjp/fed_fewshot_proto/data/SABS/sabs_CT_normalized_2"},
        # 'SABS_Superpix_2': {'data_dir': "/date/wjp/fed_fewshot_proto/data/SABS/sabs_CT_normalized_2"},
        # 'SABS_3': {'data_dir': "/date/wjp/fed_fewshot_proto/data/SABS/sabs_CT_normalized_3"},
        # 'SABS_Superpix_3': {'data_dir': "/date/wjp/fed_fewshot_proto/data/SABS/sabs_CT_normalized_3"},
        # 'SABS_4': {'data_dir': "/date/wjp/fed_fewshot_proto/data/SABS/sabs_CT_normalized_4"},
        # 'SABS_Superpix_4': {'data_dir': "/date/wjp/fed_fewshot_proto/data/SABS/sabs_CT_normalized_4"},
        }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    # observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    observer = FileStorageObserver(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
