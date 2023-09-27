"""
Validation script
"""
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

from dataloaders.dev_customized_med import med_fewshot_val
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv1 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk

from util.metric import Metric

from config_ssl_upload import ex

import tqdm
import SimpleITK as sitk
from torchvision.utils import make_grid

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        if _run.observers:
            os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
            for source_file, _ in _run.experiment_info['sources']:
                source_file = r'/home/b311/data1/wjp/Fed_fewshot_proto/config_ssl_upload.py'
                # source_file = r'/date/wjp/Fed_fewshot_proto/config_ssl_upload.py'
                os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'), exist_ok=True)
                _run.observers[0].save_file(source_file, f'source/{source_file}')
            shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Load data ######')
    ### Training set

    files_name = ['CHAOST2_Superpix_1', 'CHAOST2_Superpix_2','CHAOST2_Superpix_3','CHAOST2_Superpix_4']

    # temp_path = '/date/wjp/Fed_fewshot_proto/temp_MRI/Our_Best_S2/our-4/V{}/'.format(_config['eval_fold'])
    temp_path = '/home/b311/data1/wjp/Fed_fewshot_proto/temp_MRI/FedOurs-S1-l4/V{}/'.format(_config['eval_fold'])
    temp_txt_path = '/home/b311/data1/wjp/Fed_fewshot_proto/test_info/'
    # txt_path = temp_txt_path + 'test_reslult_4utrain(FedOurs-S2-us-2x10-4)_mri.txt'
    # txt_path = temp_txt_path + 'test_reslult_4utrain(FedOurs-S1-2x10-l4as)_mri.txt'
    # txt_path = temp_txt_path + 'test_reslult_4utrain(FedOurs-S2-2x10-l4as-3)_mri.txt'
    # txt_path = temp_txt_path + 'test_reslult_4utrain(FedOurs-S1-2x10-none-1)_mri.txt'
    txt_path = temp_txt_path + 'test_reslult_4utrain(FedOurs-S1-2x10-self-1)_mri.txt'
    vfold_result = []
    mean_Dice_Sp = []
    mean_Dice_RK = []
    mean_Dice_LK = []
    mean_Dice_Liver = []

    for sr in range(0, 8, 1):

        mean_acc = []
        for name in files_name:
            data_name = _config['dataset_CHAOST2']
            baseset_name = 'CHAOST2'
            if name=='CHAOST2_Superpix_1':
                max_label = 4
                _config["label_sets"] = 5
                _config["exclude_cls_list"] = [1]
                client_idx = 'CHAOST2_1'
                # _config["label_sets"] = 1
                # _config["exclude_cls_list"] = [2,3,4]
                # _config["label_sets"] = 9
                # _config["exclude_cls_list"] = [2, 3]
                # _config["label_sets"] = 11
                # _config["exclude_cls_list"] = [1, 2, 3, 4]
                print(f'###### Reload model {name} ######'.format(name))


            elif name=='CHAOST2_Superpix_2':
                max_label = 4
                _config["label_sets"] = 6
                _config["exclude_cls_list"] = [2]
                client_idx = 'CHAOST2_2'
                # _config["label_sets"] = 2
                # _config["exclude_cls_list"] = [1,3,4]
                # _config["label_sets"] = 10
                # _config["exclude_cls_list"] = [1, 4]
                # _config["label_sets"] = 11
                # _config["exclude_cls_list"] = [1, 2, 3, 4]

                print(f'###### Reload model {name} ######'.format(name))

            elif name=='CHAOST2_Superpix_3':
                max_label = 4
                _config["label_sets"] = 7
                _config["exclude_cls_list"] = [3]
                client_idx = 'CHAOST2_3'
                # _config["label_sets"] = 3
                # _config["exclude_cls_list"] = [1,2,4]
                # _config["label_sets"] = 10
                # _config["exclude_cls_list"] = [1, 4]
                # _config["label_sets"] = 11
                # _config["exclude_cls_list"] = [1, 2, 3, 4]
                print(f'###### Reload model {name} ######'.format(name))
            elif name=='CHAOST2_Superpix_4':
                max_label = 4
                _config["label_sets"] = 8
                _config["exclude_cls_list"] = [4]
                client_idx = 'CHAOST2_4'
                # _config["label_sets"] = 4
                # _config["exclude_cls_list"] = [1,2,3]
                # _config["label_sets"] = 9
                # _config["exclude_cls_list"] = [2, 3]
                # _config["label_sets"] = 11
                # _config["exclude_cls_list"] = [1,2,3,4]
                print(f'###### Reload model {name} ######'.format(name))
            else:
                raise ValueError(f'Dataset: {data_name} not found')


            # reload_weight = temp_path + '{}_round:{}.pth'.format(client_idx, sr)   # FedProto
            reload_weight = temp_path + '{}_{}.pth'.format(client_idx, sr)  # FedOurs


            _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
            model = FewShotSeg(pretrained_path=reload_weight, cfg=_config['model'])
            model = model.cuda()
            model.eval()

            test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]

            ### Transforms for data augmentation
            te_transforms = None

            assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

            _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
            _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

            if baseset_name == 'SABS': # for CT we need to know statistics of
                tr_parent = SuperpixelDataset( # base dataset
                    which_dataset = baseset_name,
                    base_dir=_config['path'][data_name]['data_dir'],
                    idx_split = _config['eval_fold'],
                    mode='train',
                    min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
                    transforms=None,
                    nsup = _config['task']['n_shots'],
                    scan_per_load = _config['scan_per_load'],
                    exclude_list = _config["exclude_cls_list"],
                    superpix_scale = _config["superpix_scale"],
                    fix_length = _config["max_iters_per_load"],
                    client_splite = name,
                )
                norm_func = tr_parent.norm_func
            else:
                norm_func = get_normalize_op(modality = 'MR', fids = None)


            te_dataset, te_parent = med_fewshot_val(
                dataset_name = baseset_name,
                base_dir=_config['path'][baseset_name]['data_dir'],
                idx_split = _config['eval_fold'],
                scan_per_load = _config['scan_per_load'],
                act_labels=test_labels,
                npart = _config['task']['npart'],
                nsup = _config['task']['n_shots'],
                extern_normalize_func = norm_func,
                client_splite=name,
            )

            ### dataloaders
            testloader = DataLoader(
                te_dataset,
                batch_size = 1,
                shuffle=False,
                num_workers=1,
                pin_memory=False,
                drop_last=False
            )

            _log.info('###### Set validation nodes ######')
            mar_val_metric_node = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])

            _log.info('###### Starting validation ######')
            model.eval()
            mar_val_metric_node.reset()

            with torch.no_grad():
                save_pred_buffer = {} # indexed by class

                for curr_lb in test_labels:
                    te_dataset.set_curr_cls(curr_lb)
                    support_batched = te_parent.get_support(curr_class = curr_lb, class_idx = [curr_lb], scan_idx = _config["support_idx"], npart=_config['task']['npart'])

                    # way(1 for now) x part x shot x 3 x H x W] #
                    support_images = [[shot.cuda() for shot in way]
                                        for way in support_batched['support_images']] # way x part x [shot x C x H x W]
                    suffix = 'mask'
                    support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                        for way in support_batched['support_mask']]
                    support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                        for way in support_batched['support_mask']]

                    curr_scan_count = -1 # counting for current scan
                    _lb_buffer = {} # indexed by scan

                    last_qpart = 0 # used as indicator for adding result to buffer

                    for sample_batched in testloader:

                        _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                        if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
                            continue
                        if sample_batched["is_start"]:
                            ii = 0
                            curr_scan_count += 1
                            _scan_id = sample_batched["scan_id"][0]
                            outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                            outsize = (256, 256, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                            # outsize = (257, 257, outsize[0])  # original image read by itk: Z, H, W, in prediction we use H, W, Z

                            _pred = np.zeros( outsize )
                            _pred.fill(np.nan)

                        q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
                        query_images = [sample_batched['image'].cuda()]
                        query_labels = torch.cat([ sample_batched['label'].cuda()], dim=0)

                        # [way, [part, [shot x C x H x W]]] ->
                        sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                        sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                        sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                        query_pred, _, _, assign_mats ,_,_= model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )

                        query_pred = np.array(query_pred.argmax(dim=1)[0].cpu())
                        _pred[..., ii] = query_pred.copy()

                        if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                            mar_val_metric_node.record(query_pred, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count)
                        else:
                            pass

                        ii += 1
                        # now check data format
                        if sample_batched["is_end"]:
                            if _config['dataset_CHAOST2']!= 'C0':
                                _lb_buffer[_scan_id] = _pred.transpose(2,0,1) # H, W, Z -> to Z H W
                            else:
                                _lb_buffer[_scan_id] = _pred

                    save_pred_buffer[str(curr_lb)] = _lb_buffer

                ### save results
                # for curr_lb, _preds in save_pred_buffer.items():
                #     for _scan_id, _pred in _preds.items():
                #         _pred *= float(curr_lb)
                #         itk_pred = convert_to_sitk(_pred, te_dataset.dataset.info_by_scan[_scan_id])
                #         fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_scan_id}_label_{curr_lb}_baseset_name_{baseset_name}.nii.gz')
                #         # sitk.WriteImage(itk_pred, fid, True)
                #         _log.info(f'###### {fid} has been saved ######')
                #
                # del save_pred_buffer

            del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred

            # compute dice scores by scan
            m_classDice,_, m_meanDice,_, m_rawDice = mar_val_metric_node.get_mDice(labels=sorted(test_labels), n_scan=None, give_raw = True)

            m_classPrec,_, m_meanPrec,_,  m_classRec,_, m_meanRec,_, m_rawPrec, m_rawRec = mar_val_metric_node.get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw = True)


            with open(txt_path, 'a', encoding='utf-8') as f:
                print("------------------------------------------------------------------", file=f)
                print("dataset:", baseset_name, "com_round:", sr, "weights:", reload_weight, file=f)
                print('mar_val_batches_classDice', m_classDice.tolist(), file=f)
                print('m_meanDice', m_meanDice.tolist(), file=f)
                print('mean_acc_append', mean_acc.append(m_meanDice.tolist()), file=f)

            f.close()

            if name=='CHAOST2_Superpix_1':
                mean_Dice_Liver.append(m_meanDice.tolist())
            elif name=='CHAOST2_Superpix_2':
                mean_Dice_RK.append(m_meanDice.tolist())
            elif name=='CHAOST2_Superpix_3':
                mean_Dice_LK.append(m_meanDice.tolist())
            elif name=='CHAOST2_Superpix_4':
                mean_Dice_Sp.append(m_meanDice.tolist())

            mar_val_metric_node.reset() # reset this calculation node

            # write validation result to log file
            _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
            _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
            _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

            _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
            _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
            _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

            _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
            _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
            _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

            _log.info(f'mar_val batches classDice: {m_classDice}')
            _log.info(f'mar_val batches meanDice: {m_meanDice}')

            _log.info(f'mar_val batches classPrec: {m_classPrec}')
            _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

            _log.info(f'mar_val batches classRec: {m_classRec}')
            _log.info(f'mar_val batches meanRec: {m_meanRec}')
            _log.info(f'End of validation')
            # return 1
            print(f"over datasetes:{data_name}---testinig".format(data_name))
            print("============ ============")
        with open(txt_path, 'a', encoding='utf-8') as f:
            dice_avg = sum(mean_acc) / len(mean_acc)
            print('-----------------mean-acc:{}-------------------'.format(dice_avg), file=f)
        f.close()
        vfold_result.append(dice_avg)
    print("-----------{}-----------".format(len(vfold_result)))
    with open(txt_path, 'a', encoding='utf-8') as f:
        print('-----------------max-acc:{}-------------------'.format(max(vfold_result)), file=f)
        print('-----------------max-c1-acc:{}-------------------'.format(max(mean_Dice_Liver)), file=f)
        print('-----------------max-c2-acc:{}-------------------'.format(max(mean_Dice_RK)), file=f)
        print('-----------------max-c3-acc:{}-------------------'.format(max(mean_Dice_LK)), file=f)
        print('-----------------max-c4-acc:{}-------------------'.format(max(mean_Dice_Sp)), file=f)
    f.close()