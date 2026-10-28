import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .data import iRSI, iUCMerced, iAID, iNWPU


class DatasetManager:

    def __init__(self, cfg):

        # Properties
        self.cfg = cfg

        # Dataset split setting
        self.root = cfg.DATASET.ROOT
        self.dataset_name    = cfg.DATASET.NAME
        self.num_init_cls    = cfg.DATASET.NUM_INIT_CLS
        self.num_inc_cls     = cfg.DATASET.NUM_INC_CLS
        self.num_base_shot   = cfg.DATASET.NUM_BASE_SHOT
        self.num_inc_shot    = cfg.DATASET.NUM_INC_SHOT

        # training setting of data
        self.shuffle         = cfg.DATASET.SHUFFLE
        self.seed            = cfg.SEED
        self.num_workers     = cfg.DATALOADER.NUM_WORKERS
        self.train_batchsize_base = cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE
        self.train_batchsize_inc = cfg.DATALOADER.TRAIN.BATCH_SIZE_INC
        self.test_batchsize = cfg.DATALOADER.TEST.BATCH_SIZE

        # setup data
        self._setup_data(self.root, self.shuffle, self.dataset_name)
        self.class_index_in_task = []
        self.class_index_in_task.append(np.arange(0, self.num_init_cls))
        for start in range(self.num_init_cls, self.num_total_classes, self.num_inc_cls):
            end = min(start + self.num_inc_cls, self.num_total_classes)
            self.class_index_in_task.append(np.arange(start, end))
        self.num_tasks = len(self.class_index_in_task)
        self.train_transform, self.test_transform = self._set_transform()

    def _map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    def _setup_data(self, root, shuffle, dataset_name):
        full_dataset = get_data_source(root, dataset_name)
        self.class_names = full_dataset.classes
        self.template = full_dataset.template
        self.train_data, self.train_targets = full_dataset.get_train_data()
        self.test_data, self.test_targets = full_dataset.get_test_data()

        if not isinstance(self.train_targets, np.ndarray):
            self.train_targets = np.array(self.train_targets)
        if not isinstance(self.test_targets, np.ndarray):
            self.test_targets = np.array(self.test_targets)

        order = [i for i in range(len(np.unique(self.train_targets)))]
        if shuffle:
            np.random.seed(self.seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = np.arange(0, len(self.class_names)).tolist()
        self._class_order = order
        print("seed: ", self.seed)
        print("class order: ", self._class_order)
        self.train_targets = self._map_new_class_index(
            self.train_targets, self._class_order
        )
        self.test_targets = self._map_new_class_index(self.test_targets, self._class_order)
        self.class_names = [self.class_names[i] for i in self._class_order]
        print("class name: ", self.class_names)

        self.num_total_classes = len(self.class_names)

    def get_dataset(self, task_id, source, mode=None, accumulated_past=False):
        assert 0 <= task_id < len(self.class_index_in_task), \
               f"task id {task_id} should be in range [0, {len(self.class_index_in_task) - 1}]"

        if source == 'train':
            x, y = self.train_data, self.train_targets
            if accumulated_past:
                class_idx = np.concatenate(self.class_index_in_task[0: task_id + 1])
            else:
                class_idx = self.class_index_in_task[task_id]

        elif source == 'test':
            x, y = self.test_data, self.test_targets
            class_idx = np.concatenate(self.class_index_in_task[0: task_id + 1])

        else:
            raise ValueError(f'Invalid data source :{source}')

        if mode == 'train':
            transform = self.train_transform
        elif mode == 'test':
            transform = self.test_transform
        else:
            raise ValueError(f'Invalid transform mode: {mode}')

        def find_sublist_indices(matrix, numbers):
            indices = {}
            for x in numbers:
                found = False
                for i, sublist in enumerate(matrix):
                    if x in sublist:
                        indices[x] = i
                        found = True
                        break
                if not found:
                    indices[x] = -1
            return indices

        class_to_task_id = find_sublist_indices(self.class_index_in_task, class_idx)
        num_shot = self.num_base_shot if task_id == 0 else self.num_inc_shot
        data, targets = self._select_data_from_class_index(x, y, class_idx, num_shot, source)
        task_dataset = TaskDataset(data, targets, transform, class_to_task_id, self.class_names)
        return task_dataset

    def get_dataloader(self, task_id, source, mode=None, accumulate_past=False):
        assert source in ['train', 'test'], f'data source must be in ["train", "test"], got {source}'
        if mode == None:
            mode = source
        dataset = self.get_dataset(task_id, source, mode, accumulate_past)
        if source == 'train':
            if task_id == 0:
                batchsize = self.train_batchsize_base
            else:
                batchsize = self.train_batchsize_inc
            loader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                pin_memory=True)
        elif source == 'test':
            loader = DataLoader(dataset,
                                batch_size=self.test_batchsize,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                pin_memory=True)
        else:
            raise ValueError(f'Invalid data source: {source}')
        return loader

    def _select_data_from_class_index(self, x, y, class_idx, shot, source):
        ret_x = []
        ret_y = []
        if isinstance(x, list):
            x = np.array(x)
        for c in class_idx:
            idx_c = np.where(y == c)[0]

            if shot is not None and source == 'train':
                if shot == -1:
                    idx_selected = idx_c

                elif shot > len(idx_c):
                    print(f'shot:{shot} is greater than num of sample:{len(idx_c)} in class{c}')
                    idx_selected = idx_c
                else:
                    idx_selected = np.random.choice(idx_c, size=shot, replace=False)
            else:
                idx_selected = idx_c

            ret_x.append(x[idx_selected])
            ret_y.append(y[idx_selected])
        ret_x = np.concatenate(ret_x)
        ret_y = np.concatenate(ret_y)

        return ret_x, ret_y

    def _set_transform(self):
        img_size = 224
        MEAN = [0.48145466, 0.4578275, 0.40821073]
        STD  = [0.26862954, 0.26130258, 0.27577711]
        train_transform  = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return train_transform, test_transform


class TaskDataset(Dataset):
    def __init__(self, images, labels, transform, class_to_task_id=None, class_name=None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.transform = transform
        self.use_path = isinstance(images[0], str)
        self.class_to_task_id = class_to_task_id
        self.class_name = class_name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.transform(pil_loader(self.images[idx]))
        else:
            image = self.transform(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        if self.class_to_task_id is not None:
            task_id = self.class_to_task_id[label]
        else:
            task_id = -1

        if self.class_name is not None:
            cls_name = self.class_name[label]
        else:
            cls_name = ''

        ret = {
            'idx': idx,
            'image': image,
            'label': label,
            'cls_name': cls_name,
            'task_id': task_id
        }
        return ret


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_data_source(root, name):
    source_dict = {
        'nwpu': iNWPU,
        'iucmerced': iUCMerced,
        'irsi': iRSI,
        'iaid': iAID,
    }
    return source_dict[name.lower()](root=root)
