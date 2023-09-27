"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
import copy
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric

from config_ssl_upload import ex
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.figure()
# config pre-trained model caching path
# os.environ['TORCH_HOME'] = "/data1/wjp/fed_fewshot_proto/pretrained_model"
os.environ['TORCH_HOME'] = "/data1/wjp/Fed_fewshot_proto/pretrained_model"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def slices_agg(slices_protos,iters):
    slices_avg = slices_protos/iters
    return slices_avg
def epoch_agg(slices_sum,epoch):
    epoch_proto = slices_sum/epoch
    return epoch_proto
def dynamic_head(name):
    if name == 'CHAOST2_Superpix_1':
        head = 0
    elif name == 'CHAOST2_Superpix_2':
        head = 1
    elif name == 'CHAOST2_Superpix_3':
        head = 2
    elif name == 'CHAOST2_Superpix_4':
        head =3
    return head


@ex.automain
def main(_run, _config, _log):

    if _run.observers:
        try:
            os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        except OSError:
            pass

        for source_file, _ in _run.experiment_info['sources']:
            source_file = r'/data1/wjp/Fed_fewshot_proto/config_ssl_upload.py'

            try:
                os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),exist_ok=True)
            except OSError:
                pass
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        # shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    train_loss_lv = []
    train_loss_rk = []
    train_loss_lk = []
    train_loss_sp = []


    rounds = 10 #[20,10,5]
    step = 2000 #[1000,2000,4000]
    snap = 12  # running snaps
    T = 0.1
    t = 1 # snap-1  [snap1(0.1-0.60),snap2(1-0.649),snap3(10-0.62),snap4(5-0.633),snap5(1-lr-0.658),snap6(2-0.624),snap7(3-lr-0.60),snap8(1-(lr-10)-0.6635),snap9(1-lr(20-15)-0.650),]
    #[2x10:snap10(1-lr2-0.59),snap10(1-lr20-0.648),snap11(0.1-lr20-x),snap11(1-lr10-0.58),snap11(1-lr30-0.658),snap12()]
    for i in range(rounds):
        if i==0:
            try:
                para = torch.load("/data1/wjp/Fed_fewshot_proto/mri-global-protos/glo_proto.pth")
                print("last glob-proto is exist!")
                os.remove("/data1/wjp/Fed_fewshot_proto/mri-global-protos/glo_proto.pth")
                print("remove success!")
            except:
                pass
        _log.info(f'###### round:{i} of {rounds}######'.format(i,rounds))
        ### Training set

        files_name = ['CHAOST2_Superpix_1', 'CHAOST2_Superpix_2', 'CHAOST2_Superpix_3', 'CHAOST2_Superpix_4']
        # files_name = ['CHAOST2_Superpix_1','CHAOST2_Superpix_2','CHAOST2_Superpix_3','CHAOST2_Superpix_4']

        epoch_proto = []

        for name in files_name:
            dynamic_lr = [ (ii + 1) * 1000 for ii in range(((5)*step) // 1000 - 1)]
            if name=='CHAOST2_Superpix_1':
                print(f"nowing running:{name}".format(name))
                data_name = _config['dataset_1']
                baseset_name = 'CHAOST2_1'
                head = 0
                _config["label_sets"] = 1
                _config["exclude_cls_list"] = [2,3,4]
                try:
                    torch.load("/data1/wjp/Fed_fewshot_proto/examples/exps/myexperiments-ours_MIDDLE/mySSL_train__vfold0_CHAOST2_Superpix_sets__1shot/{}/snapshots/{}+CHAOST2_1+round:{}.pth".format(snap,step,i - 1))
                    reload_pretrain_path = "/data1/wjp/Fed_fewshot_proto/examples/exps/myexperiments-ours_MIDDLE/mySSL_train__vfold0_CHAOST2_Superpix_sets__1shot/{}/snapshots/{}+CHAOST2_1+round:{}.pth".format(snap,step,i - 1)
                    print("-----------------local para has been loaded-----------------")
                except:
                    reload_pretrain_path=None
                    print("-----------------now inital local para-----------------")
            elif name == 'CHAOST2_Superpix_2' :
                print(f"nowing running:{name}".format(name))
                data_name = _config['dataset_2']
                baseset_name = 'CHAOST2_2'
                head = 1
                _config["label_sets"] = 2
                _config["exclude_cls_list"] = [1,3,4]
                try:
                    torch.load("/data1/wjp/Fed_fewshot_proto/examples/exps/myexperiments-ours_MIDDLE/mySSL_train__vfold0_CHAOST2_Superpix_sets__1shot/{}/snapshots/{}+CHAOST2_2+round:{}.pth".format(snap,step,i - 1))
                    reload_pretrain_path = "/data1/wjp/Fed_fewshot_proto/examples/exps/myexperiments-ours_MIDDLE/mySSL_train__vfold0_CHAOST2_Superpix_sets__1shot/{}/snapshots/{}+CHAOST2_2+round:{}.pth".format(snap,step,i - 1)
                    print("-----------------local para has been loaded-----------------")
                except:
                    reload_pretrain_path=None
                    print("-----------------now inital local para-----------------")
            elif name == 'CHAOST2_Superpix_3' :
                print(f"nowing running:{name}".format(name))
                data_name = _config['dataset_3']
                baseset_name = 'CHAOST2_3'
                head = 2
                _config["label_sets"] = 3
                _config["exclude_cls_list"] = [1,2,4]
                try:
                    torch.load("/data1/wjp/Fed_fewshot_proto/examples/exps/myexperiments-ours_MIDDLE/mySSL_train__vfold0_CHAOST2_Superpix_sets__1shot/{}/snapshots/{}+CHAOST2_3+round:{}.pth".format(snap,step,i - 1))
                    reload_pretrain_path = "/data1/wjp/Fed_fewshot_proto/examples/exps/myexperiments-ours_MIDDLE/mySSL_train__vfold0_CHAOST2_Superpix_sets__1shot/{}/snapshots/{}+CHAOST2_3+round:{}.pth".format(snap,step,i - 1)
                    print("-----------------local para has been loaded-----------------")
                except:
                    reload_pretrain_path=None
                    print("-----------------now inital local para-----------------")
            elif name == 'CHAOST2_Superpix_4' :
                print(f"nowing running:{name}".format(name))
                data_name = _config['dataset_4']
                baseset_name = 'CHAOST2_4'
                head = 3
                _config["label_sets"] = 4
                _config["exclude_cls_list"] = [1,2,3]
                try:
                    torch.load("/data1/wjp/Fed_fewshot_proto/examples/exps/myexperiments-ours_MIDDLE/mySSL_train__vfold0_CHAOST2_Superpix_sets__1shot/{}/snapshots/{}+CHAOST2_4+round:{}.pth".format(snap,step,i - 1))
                    reload_pretrain_path = "/data1/wjp/Fed_fewshot_proto/examples/exps/myexperiments-ours_MIDDLE/mySSL_train__vfold0_CHAOST2_Superpix_sets__1shot/{}/snapshots/{}+CHAOST2_4+round:{}.pth".format(snap,step,i - 1)
                    print("-----------------local para has been loaded-----------------")
                except:
                    reload_pretrain_path=None
                    print("-----------------now inital local para-----------------")

            set_seed(_config['seed'])
            cudnn.enabled = True
            cudnn.benchmark = True
            torch.cuda.set_device(device=_config['gpu_id'])
            torch.set_num_threads(1)

            _log.info('###### Create model ######')
            model = FewShotSeg(pretrained_path=reload_pretrain_path, cfg=_config['model'])
            model = model.cuda()
            model.train()

            ## Transforms for data augmentation
            tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
            assert _config['scan_per_load'] < 0  # by default we load the entire dataset directly

            test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
            _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
            _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

            tr_parent = SuperpixelDataset( # base dataset
                which_dataset = baseset_name,
                base_dir=_config['path'][data_name]['data_dir'],
                idx_split = _config['eval_fold'],
                mode='train',
                min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
                transforms=tr_transforms,
                nsup = _config['task']['n_shots'],
                scan_per_load = _config['scan_per_load'],
                exclude_list = _config["exclude_cls_list"],
                superpix_scale = _config["superpix_scale"],
                fix_length = _config["max_iters_per_load"] if (data_name == 'CHAOST2_Superpix_1')or(data_name == 'CHAOST2_Superpix_2') or(data_name == 'CHAOST2_Superpix_3')or(data_name == 'CHAOST2_Superpix_4')else None
            )

            ### dataloaders
            trainloader = DataLoader(
                tr_parent,
                batch_size=_config['batch_size'],
                shuffle=True,
                num_workers=_config['num_workers'],
                pin_memory=True,
                drop_last=True
            )

            _log.info('###### Set optimizer ######')
            if _config['optim_type'] == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
            else:
                raise NotImplementedError

            scheduler = MultiStepLR(optimizer, milestones=dynamic_lr,  gamma = _config['lr_step_gamma'])
            # scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=_config['lr_step_gamma'])

            my_weight = compose_wt_simple(_config["use_wce"], data_name)
            criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

            i_iter = 0 # total number of iteration

            n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

            log_loss = {'query_loss': 0,'loss': 0, 'align_loss': 0}

            _log.info('###### Training ######')

            for sub_epoch in range(n_sub_epoches):
                _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
                sum_proto = torch.zeros(1,256).cuda()

                num_iter = 0
                for _, sample_batched in enumerate(trainloader):
                    # Prepare input
                    i_iter += 1

                    # add writers
                    support_images = [[shot.cuda() for shot in way]
                                      for way in sample_batched['support_images']]
                    support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]

                    query_images = [query_image.cuda()
                                    for query_image in sample_batched['query_images']]
                    query_labels = torch.cat(
                        [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

                    optimizer.zero_grad()
                    # FIXME: in the model definition, filter out the failure case where pseudolabel falls outside of image or too small to calculate a prototype


                    try:
                        query_pred, align_loss, debug_vis, assign_mats,fg_proto,bg_proto = model(support_images, support_fg_mask, support_bg_mask, query_images,isval = False, val_wsize = None)


                        proto = fg_proto
                        sum_proto = sum_proto + fg_proto

                    except:
                        print('Faulty batch detected, skip')
                        continue

                    try:
                        global_protos = torch.load("/data1/wjp/Fed_fewshot_proto/mri-global-protos/glo_proto.pth")
                         # SimClr contrastive
                        pos_proto = global_protos[head]

                        sim_1 =torch.exp(F.cosine_similarity(proto,pos_proto,dim=1)/T)

                        sim_2 = torch.exp(F.cosine_similarity(proto,global_protos[0],dim=1)/T)+torch.exp(F.cosine_similarity(proto,global_protos[1],dim=1)/T) + torch.exp(F.cosine_similarity(proto,global_protos[2],dim=1)/T)+torch.exp(F.cosine_similarity(proto,global_protos[3],dim=1)/T)

                        loss_con = -torch.log((sim_1) / (sim_2))
                        loss_con = loss_con.item()

                        if i_iter == 1:
                            print("---------------------Has beed obtained global proto--------------------")

                    except:
                        if i_iter==1:
                            print("----------------------initial_global_proto------------------------")
                        loss_con = 0




                    query_loss = criterion(query_pred, query_labels)

                    loss = query_loss + align_loss + t * loss_con
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Log loss

                    query_loss = query_loss.detach().data.cpu().numpy()
                    align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
                    # loss_con = loss_con.detach().data.cpu().numpy()
                    loss = loss.detach().data.cpu().numpy()
                    _run.log_scalar('query_loss', query_loss)
                    _run.log_scalar('align_loss', align_loss)
                    _run.log_scalar('loss', loss)
                    log_loss['query_loss'] += query_loss
                    log_loss['align_loss'] += align_loss
                    log_loss['loss'] += loss

                    # print loss and take snapshots
                    if (i_iter + 1) % _config['print_interval'] == 0:

                        query_loss = log_loss['query_loss'] / _config['print_interval']
                        align_loss = log_loss['align_loss'] / _config['print_interval']
                        loss = log_loss['loss'] / _config['print_interval']

                        if name == 'CHAOST2_Superpix_1':
                            train_loss_lv.append(loss)
                        elif name == 'CHAOST2_Superpix_2':
                            train_loss_rk.append(loss)
                        elif name == 'CHAOST2_Superpix_3':
                            train_loss_lk.append(loss)
                        elif name == 'CHAOST2_Superpix_4':
                            train_loss_sp.append(loss)

                        log_loss['query_loss'] = 0
                        log_loss['align_loss'] = 0
                        log_loss['loss'] = 0

                        print(f'step {i_iter + 1}: query_loss: {query_loss}, align_loss: {align_loss}, loss: {loss}')


                    if (i_iter + 1) % 2000 == 0:
                        _log.info('###### Taking snapshot ######')
                        torch.save(model.state_dict(),
                                   os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}+{baseset_name}+round:{i}.pth'))

                    if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix_1'or data_name == 'CHAOST2_Superpix_2'or data_name == 'CHAOST2_Superpix_3'or data_name == 'CHAOST2_Superpix_4':
                        if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                            _log.info('###### Reloading dataset ######')
                            trainloader.dataset.reload_buffer()
                            print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

                    if (i_iter - 2) > _config['n_steps']:
                        return 1 # finish up

                # sum_proto = copy(sum_proto)
                slices_avg = slices_agg(sum_proto,float(i_iter))

            epoch_proto.append(slices_avg)

            print(len(epoch_proto),"global_proto_length")

        torch.save(epoch_proto, "/data1/wjp/Fed_fewshot_proto/mri-global-protos/glo_proto.pth")
        print("----------------round:{} is finish!-------------".format(i))

    plt.plot(train_loss_lv, 'b', label="Liver_train_loss")
    plt.plot(train_loss_lk, 'r', label="LK_train_loss")
    plt.plot(train_loss_rk, 'g', label="RK_train_loss")
    plt.plot(train_loss_sp, 'y', label="Spleen_train_loss")
    plt.ylabel('loss')
    plt.xlabel('step_iter')
    plt.legend()
    plt.savefig(os.path.join("/data1/wjp/Fed_fewshot_proto/examples/cl-logs", "MRI_train_loss.jpg"))


