import torch
import numpy as np
from sklearn.metrics import confusion_matrix

class AccuracyEvaluator:
    
    def __init__(self, class_index_per_task):
        #任务分割的类别索引列表
        self.class_index_per_task = class_index_per_task
        self.num_tasks = len(class_index_per_task)


    def confusion_matrix(self, logits, targets, task_id, normalize=False):
        class_conf_matrix = self._class_wise_confusion_matrix(logits, targets, task_id, normalize)
        task_conf_matrix = self._task_wise_confusion_matrix(logits, targets, task_id, normalize)

        return {'class_conf_matrix': class_conf_matrix,
                'task_conf_matrix': task_conf_matrix}


    def calc_accuracy(self, logits, targets, task_id):
        logits = logits.cpu().numpy()
        targets = targets.cpu().numpy()

        #样本中预测正确的数量
        overall_right_cnt = self._count_right_pred_num(logits, targets)
        #得到预测准确率
        overall_acc_mean = overall_right_cnt / len(targets)

        #截止目前任务为止，见过的所有任务的类别，列表形式
        seen_task_classes = self.class_index_per_task[:task_id + 1]
        task_accuracies = []
        for classes in seen_task_classes:
            #获取属于当前类别对应的targets的任务id
            task_sample_indices = np.where(np.isin(targets, classes))[0]
            if len(task_sample_indices) == 0:
                task_accuracies.append(0.0)
                continue

            task_sample_logits = logits[task_sample_indices]
            task_sample_targets = targets[task_sample_indices]

            task_right_cnt = self._count_right_pred_num(task_sample_logits, task_sample_targets)

            task_acc_mean = task_right_cnt / len(task_sample_indices)
            #获取任务级别的准确率
            task_accuracies.append(round(100 * task_acc_mean, 2))

        return {'mean_acc': round(100 * overall_acc_mean, 2), 
                'task_acc': task_accuracies}


    def _count_right_pred_num(self, logits, targets):
        pred = np.argmax(logits, axis=1)
        return np.sum(pred == targets)


    def _determine_tasks(self, samples, task_classes):
        tasks = np.zeros_like(samples)
        for task_id, classes in enumerate(task_classes):
            #确保samples中的类别属于当前任务的类别集合
            class_mask = np.isin(samples, classes)
            #每一个样本对应一个任务ID
            tasks[class_mask] = task_id
        return tasks
    


    def _task_wise_confusion_matrix(self, logits, targets, task_id, normalize=False):
        logits_np = logits.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        seen_task_classes = [cls for cls in self.class_index_per_task[:task_id + 1]]
        actual_tasks = self._determine_tasks(targets_np, seen_task_classes)
        predicted_tasks = self._determine_tasks(np.argmax(logits_np, axis=1), seen_task_classes)

        task_conf_matrix = confusion_matrix(actual_tasks, predicted_tasks, labels=range(len(seen_task_classes)))

        if normalize:
            task_conf_matrix = task_conf_matrix.astype('float')
            row_sums = task_conf_matrix.sum(axis=1, keepdims=True)
            task_conf_matrix /= row_sums

        return task_conf_matrix

    def _class_wise_confusion_matrix(self, logits, targets, task_id, normalize=False):
        logits_np = logits.cpu().numpy()
        targets_np = targets.cpu().numpy()

        #取得到当前任务及之前任务的所有类别
        seen_classes = np.concatenate([cls for cls in self.class_index_per_task[:task_id + 1]])
        #避免重复
        unique_seen_classes = np.unique(seen_classes)
        #过滤出属于这些类别的样本，不属于学习过的类别的数据被标记为false
        valid_indices = np.isin(targets_np, unique_seen_classes)
        valid_logits = logits_np[valid_indices]
        valid_targets = targets_np[valid_indices]

        preds = np.argmax(valid_logits, axis=1)
        #计算混淆矩阵，预测结果与真是标签的对比
        conf_matrix = confusion_matrix(valid_targets, preds, labels=unique_seen_classes)

        #是否进行归一化
        if normalize:
            conf_matrix = conf_matrix.astype('float')
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix /= row_sums

        return conf_matrix


    # def task_class_confusion_matrix(self, class_labels, true_task_labels, logits):
    #     """
    #     Compute the task-class confusion matrix.

    #     Args:
    #     - class_labels (torch.Tensor): Tensor of ground truth class labels for each sample.
    #     - true_task_labels (torch.Tensor): Tensor of ground truth task labels for each sample.
    #     - logits (torch.Tensor): The logits output from the model for each sample.
        
    #     Returns:
    #     - np.array: A confusion matrix of shape (num_classes, num_tasks)
    #     """
    #     if isinstance(class_labels, torch.Tensor):
    #         class_labels = class_labels.cpu().numpy()
    #     if isinstance(true_task_labels, torch.Tensor):
    #         true_task_labels = true_task_labels.cpu().numpy()
    #     if isinstance(logits, torch.Tensor):
    #         logits = logits.cpu().numpy()
        
    #     predicted_task_labels = np.argmax(logits, axis=1)
        
    #     unique_classes = np.unique(class_labels)
    #     unique_tasks = np.arange(10)
        
    #     confusion_mat = np.zeros((len(unique_classes), len(unique_tasks)))
        
    #     for i, cls in enumerate(unique_classes):
    #         for j, task in enumerate(unique_tasks):
    #             idx = np.where((class_labels == cls) & (predicted_task_labels == task))[0]
    #             task_correct = np.sum(predicted_task_labels[idx] == task)
    #             confusion_mat[i, j] = task_correct
        
    #     return confusion_mat