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
from dataloaders.GenericSuperDatasetv1 import SuperpixelDataset
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
os.environ['TORCH_HOME'] = "./pretrained_model"

# @ex.capture()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




@ex.automain
def main(_run, _config, _log):

    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            source_file = r'/home/b311/data1/wjp/Fed_fewshot_proto/config_ssl_upload.py'
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'), exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    train_loss_lv = []
    train_loss_rk = []
    train_loss_lk = []
    train_loss_sp = []

    # temp_path = '/home/b311/data1/wjp/Fed_fewshot_proto/temp_CT/FedOurs-none-S1/V{}/'.format(_config['eval_fold'])
    temp_path = '/home/b311/data1/wjp/Fed_fewshot_proto/temp_CT/FedOurs-S1-l4/V{}/'.format(_config['eval_fold'])
    temp_proto_path = '/home/b311/data1/wjp/Fed_fewshot_proto/temp_space/temp_cl_ct/'
    rounds = _config['rounds']  # [20,10,5]
    step = _config['n_steps'] #[1000,2000,4000]
    T = 0.1
    t = 1

    for i in range(rounds):
        _log.info('###### Load data ######')
        ### Training set
        if i == 0:
            if os.listdir(temp_path) != []:
                exit()
            try:
                para = torch.load(temp_proto_path + "glo_proto.pt")
                print("last glob-proto is exist!")
                os.remove(temp_proto_path + "glo_proto.pt")
                print("remove success!")
            except:
                pass


        files_name = ['SABS_Superpix_1', 'SABS_Superpix_2', 'SABS_Superpix_3', 'SABS_Superpix_4']


        glb_protos = []
        clients_w = []
        for name in files_name:
            # if i>=5:
            #     j = 7
            # else:
            #     j = 4 + i

            dynamic_lr = [(ii + 1) * 1000 for ii in range((rounds * step) // 1000 - 1)]
            data_name = _config['dataset_SABS']
            baseset_name = 'SABS'
            if name=='SABS_Superpix_1':
                print(f"nowing running:{name}".format(name))
                head = 0
                _config["label_sets"] = 1
                # _config["exclude_cls_list"] = [2,3,6]
                client_idx = 'SABS_1'
            elif name == 'SABS_Superpix_2' :
                print(f"nowing running:{name}".format(name))
                head = 1
                _config["label_sets"] = 2
                # _config["exclude_cls_list"] = [1,3,6]
                client_idx = 'SABS_2'
            elif name == 'SABS_Superpix_3' :
                print(f"nowing running:{name}".format(name))
                head = 2
                _config["label_sets"] = 3
                # _config["exclude_cls_list"] = [1,2,6]
                client_idx = 'SABS_3'
            elif name == 'SABS_Superpix_4' :
                print(f"nowing running:{name}".format(name))
                head = 3
                _config["label_sets"] = 4
                # _config["exclude_cls_list"] = [1,2,3]
                client_idx = 'SABS_4'

            '''setting 1'''
            _config["exclude_cls_list"] = [ ]

            set_seed(_config['seed'])
            cudnn.enabled = True
            cudnn.benchmark = True
            torch.cuda.set_device(device=_config['gpu_id'])
            torch.set_num_threads(1)

            try:
                torch.load(temp_path + "{}_{}.pth".format(client_idx, i-1))
                reload_pretrian_path = temp_path + "{}_{}.pth".format(client_idx, i-1)
            except:
                reload_pretrian_path = None

            _log.info('###### Create model ######')
            model = FewShotSeg(pretrained_path=reload_pretrian_path, cfg=_config['model'])
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
                fix_length=_config["max_iters_per_load"],
                client_splite=name,
                # fix_length = _config["max_iters_per_load"] if (data_name == 'CHAOST2_Superpix_1')or(data_name == 'CHAOST2_Superpix_2') or(data_name == 'CHAOST2_Superpix_3')or(data_name == 'CHAOST2_Superpix_4')else None
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

            local_proto = []
            if i != 0:
                global_protos = torch.load(temp_proto_path + "glo_proto.pt")
                # print(len(global_protos))
                # print(len(global_protos[0]),len(global_protos[1]),len(global_protos[2]),len(global_protos[3]))
                ab_proto_0 = (sum(global_protos[0]) / len(global_protos[0])).cuda()
                ab_proto_1 = (sum(global_protos[1]) / len(global_protos[1])).cuda()
                ab_proto_2 = (sum(global_protos[2]) / len(global_protos[2])).cuda()
                ab_proto_3 = (sum(global_protos[3]) / len(global_protos[3])).cuda()
            # return 1
            for sub_epoch in range(n_sub_epoches):
                _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
                for _, sample_batched in enumerate(trainloader):
                    # Prepare input
                    i_iter += 1


                    # add writers\
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
                    except:
                        print('Faulty batch detected, skip')
                        continue

                    proto = fg_proto
                    local_proto.append(fg_proto)

                    if i != 0:
                        # global_protos1 = torch.load(temp_proto_path + "glo_proto.pt")
                        # SimClr contrastive
                        pos_proto = global_protos[head]
                        # print(iter())
                        sim_1 = torch.exp(F.cosine_similarity(proto, pos_proto[i_iter-1], dim=1) / T)

                        if head == 0:
                            sim_2 = torch.exp(F.cosine_similarity(proto, pos_proto[i_iter-1], dim=1) / T) + torch.exp(
                                F.cosine_similarity(proto, ab_proto_1, dim=1) / T) + \
                                    torch.exp(F.cosine_similarity(proto, ab_proto_2, dim=1) / T) + torch.exp(
                                F.cosine_similarity(proto, ab_proto_3, dim=1) / T)
                        elif head == 1:
                            sim_2 = torch.exp(F.cosine_similarity(proto, ab_proto_0, dim=1) / T) + torch.exp(
                                F.cosine_similarity(proto, pos_proto[i_iter-1], dim=1) / T) + \
                                    torch.exp(F.cosine_similarity(proto, ab_proto_2, dim=1) / T) + torch.exp(
                                F.cosine_similarity(proto, ab_proto_3, dim=1) / T)
                        elif head == 2:
                            sim_2 = torch.exp(F.cosine_similarity(proto, ab_proto_0, dim=1) / T) + torch.exp(
                                F.cosine_similarity(proto, ab_proto_1, dim=1) / T) + \
                                    torch.exp(F.cosine_similarity(proto, pos_proto[i_iter-1], dim=1) / T) + torch.exp(
                                F.cosine_similarity(proto, ab_proto_3, dim=1) / T)
                        elif head == 3:
                            sim_2 = torch.exp(F.cosine_similarity(proto, ab_proto_0, dim=1) / T) + torch.exp(
                                F.cosine_similarity(proto, ab_proto_1, dim=1) / T) + \
                                    torch.exp(F.cosine_similarity(proto, ab_proto_2, dim=1) / T) + torch.exp(
                                F.cosine_similarity(proto, pos_proto[i_iter-1], dim=1) / T)

                        loss_con = - torch.log((sim_1) / (sim_2))
                        loss_con = loss_con.item()
                        if i_iter == 1:
                            print("---------------------Has beed obtained global proto--------------------")

                    else:
                        if i_iter == 1:
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

                        if name == 'SABS_Superpix_1':
                            train_loss_sp.append(loss)
                        elif name == 'SABS_Superpix_2':
                            train_loss_rk.append(loss)
                        elif name == 'SABS_Superpix_3':
                            train_loss_lk.append(loss)
                        elif name == 'SABS_Superpix_4':
                            train_loss_lv.append(loss)

                        log_loss['query_loss'] = 0
                        log_loss['align_loss'] = 0
                        log_loss['loss'] = 0

                        print(f'step {i_iter + 1}: query_loss: {query_loss}, align_loss: {align_loss}, loss: {loss}')


                    # if (i_iter + 1) % step == 0:
                    #     _log.info('###### Taking snapshot ######')
                    #     torch.save(model.state_dict(),
                    #                os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}+{baseset_name}+round:{i}.pth'))

                    if 'C0' in data_name or 'CHAOST2' in data_name or 'SABS' in data_name:
                        if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                            _log.info('###### Reloading dataset ######')
                            trainloader.dataset.reload_buffer()
                            print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

                    if (i_iter - 2) > _config['n_steps']:
                        return 1

            glb_protos.append(local_proto)
            print("--------glb_protos hes been:{}---------".format(len(glb_protos)))
            # print(len(epoch_proto),"global_proto_length")
            net_w = model.state_dict()
            clients_w.append(net_w)

        torch.save(glb_protos, temp_proto_path + "glo_proto.pt")
        print("----------------round:{} is finish!-------------".format(i))

        local_weights = copy.deepcopy(clients_w)
        num_sum = 23.0
        c1_num = 6.0
        c2_num = 6.0
        c3_num = 6.0
        c4_num = 5.0

        print("------------FedAvg-start!-----------")
        w1 = local_weights[0]
        w2 = local_weights[1]
        w3 = local_weights[2]
        w4 = local_weights[3]
        w_1 = {}
        w_2 = {}
        w_3 = {}
        w_4 = {}
        num = 0
        for (key1, values1), (key2, values2), (key3, values3), (key4, values4) in zip(w1.items(), w2.items(),
                                                                                      w3.items(), w4.items()):
            num += 1
            #none
            # w_1[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            # w_2[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            # w_3[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            # w_4[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum

            # aspp
            # if 'backbone' in key1:
            #     w_1[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_2[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_3[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_4[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            # else:
            #     w_1[key1] = values1
            #     w_2[key1] = values2
            #     w_3[key1] = values3
            #     w_4[key1] = values4

            # aspp+l4
            # if 'layer4' in key1 or 'layer3' in key1 or'backbone' not in key1:
            #     w_1[key1] = values1
            #     w_2[key1] = values2
            #     w_3[key1] = values3
            #     w_4[key1] = values4
            # else:
            #     w_1[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_2[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_3[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_4[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum


            # aspp+l4
            # if 'layer4' in key1 or'backbone' not in key1:
            #     w_1[key1] = values1
            #     w_2[key1] = values2
            #     w_3[key1] = values3
            #     w_4[key1] = values4
            # else:
            #     w_1[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_2[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_3[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum
            #     w_4[key1] = (values1 * c1_num + values2 * c2_num + values3 * c3_num + values4 * c4_num) / num_sum

            # self-trainning
            w_1[key1] = values1
            w_2[key1] = values2
            w_3[key1] = values3
            w_4[key1] = values4
        # print("---------------model_nums:{}--------------".format(num))
        torch.save(w_1, temp_path + "SABS_1_{}.pth".format(i))
        torch.save(w_2, temp_path + "SABS_2_{}.pth".format(i))
        torch.save(w_3, temp_path + "SABS_3_{}.pth".format(i))
        torch.save(w_4, temp_path + "SABS_4_{}.pth".format(i))
        print("-----------------------FedPer-finish!------------------")
        # return epoch_proto
    # plt.plot(train_loss_sp, 'b', label="Spleen_train_loss")
    # plt.plot(train_loss_rk, 'r', label="RK_train_loss")
    # plt.plot(train_loss_lk, 'g', label="LK_train_loss")
    # plt.plot(train_loss_lv, 'y', label="Liver_train_loss")
    # plt.ylabel('loss')
    # plt.xlabel('step_iter')
    # plt.legend()
    # plt.savefig(os.path.join("/data1/wjp/Fed_fewshot_proto/examples/cl-logs", "CT_Train_loss.jpg"))


