import torch
import torch.nn as nn
from data.data_manager import DatasetManager
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from model.ours import ZFCIL
import numpy as np
import time
from torch.utils.data import DataLoader


class Runner:

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DatasetManager(cfg,)
        self.device = cfg.DEVICE.DEVICE_NAME

        self.model = ZFCIL(cfg, self.data_manager.template, self.device)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.is_distributed = True
        else:
            self.is_distributed = False

        self.acc_list = []
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)

    def merge_dicts(self, dict_list):
        result = {}

        keys_to_merge = [
            'description_proto',
            'description_features',
            'description_targets',
            'text_features',
            'text_targets',
            'yuan_proto',
            'image_proto',
            'images_features',
            'images_targets',
        ]

        for key in keys_to_merge:
            result[key] = torch.cat([d[key] for d in dict_list], dim=0)
        result['class_index'] = [d['class_index'] for d in dict_list]

        weights = [len(d['class_index']) for d in dict_list]

        cov_keys = [
            'cov_image',
        ]
        cov_sums = {key: torch.zeros_like(dict_list[0][key]) for key in cov_keys}
        weight_sum = sum(weights)

        for i, d in enumerate(dict_list):
            for key in cov_keys:
                cov_sums[key] += d[key] * weights[i]

        for key in cov_keys:
            if weight_sum > 0:
                result[key] = cov_sums[key] / weight_sum

        return result

    @torch.no_grad()
    def run(self):
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')
        state_dict_list = []
        self.total_classes = 0
        for i in range(self.data_manager.num_tasks):

            self.total_classes += len(self.data_manager.class_index_in_task[i])
            print("self.total_classes :", self.total_classes)
            self.model.clip_model.update_fc(self.total_classes)
            self.model.eval()

            current_class_name = np.array(self.data_manager.class_names)[self.data_manager.class_index_in_task[i]]

            loader = self.data_manager.get_dataloader(i, source='train', mode='test', accumulate_past=False)

            self.train_loader_for_protonet = DataLoader(loader, batch_size=self.cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE, shuffle=True, num_workers=self.cfg.DATALOADER.NUM_WORKERS)

            current_state_dict = self.model.build_task_statistics(current_class_name, loader,
                                                                  class_index=self.data_manager.class_index_in_task[i],
                                                                  calibrate_novel_vision_proto=self.cfg.TRAINER.ZFCIL.VISION_CALIBRATION,)

            state_dict_list.append(current_state_dict)
            merged_state_dict = self.merge_dicts(state_dict_list)

            self.replace_fc(self.train_loader_for_protonet, self.model, merged_state_dict)

            start_time = time.time()
            acc = self.inference_task_covariance(i, merged_state_dict)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'+++++++++++  task {i}, time: {elapsed_time} ++++++++++++++++')

            print(f'=> Task [{i}], Acc: {acc["mean_acc"]:.3f}')
            self.acc_list.append(round(acc["mean_acc"], 3))

        print(f'Final acc:{self.acc_list}')

    @torch.no_grad()
    def inference_task_covariance(self, task_id, state_dict):

        beta = self.cfg.DATASET.BETA

        image_proto = state_dict['image_proto']
        cov_image = state_dict['cov_image']
        image_features = state_dict['images_features']
        text_features = state_dict['text_features']
        description_proto = state_dict['description_proto']
        description_features = state_dict['description_features']
        description_targets = state_dict['description_targets']

        num_base_class = len(self.data_manager.class_index_in_task[0])
        num_accumulated_class = 0
        for i in range(task_id + 1):
            num_accumulated_class += len(self.data_manager.class_index_in_task[i])
        print('num_accumulate :', num_accumulated_class)

        test_loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        all_logits = []
        all_cov_logits = []
        all_targets = []

        for i, batch in enumerate(tqdm(test_loader)):
            data, targets = self.parse_batch(batch)
            with torch.no_grad():
                logits, cov_logits = self.model.forward_ours(data, num_accumulated_class, num_base_class,
                                                             image_proto,
                                                             cov_image,
                                                             description_proto,
                                                             description_features,
                                                             description_targets,
                                                             image_features,
                                                             text_features,
                                                             beta=beta)

            all_logits.append(logits)
            all_cov_logits.append(cov_logits)
            all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id)
        eval_cov_acc = self.evaluator.calc_accuracy(torch.cat(all_cov_logits, dim=0), all_targets, task_id)
        acc = eval_cov_acc['mean_acc']
        print(f"final_acc:{acc},Test acc mean: {eval_acc['mean_acc']},tast cov acc mean :{eval_cov_acc['mean_acc'] }")
        eval_acc['mean_acc'] = acc
        return eval_acc

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets

    def replace_fc(self, trainloader, model, state_dict):
        self.model.eval()
        embedding_list = state_dict['image_proto']

        images_targets = state_dict['images_targets']

        label_list = images_targets.cpu().numpy().tolist()
        class_list = np.unique(label_list)
        for i in range(len(class_list)):
            self.model.clip_model.fc.weight.data[i] = embedding_list[i]
        return model
