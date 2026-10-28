import os
import numpy as np
from torchvision import datasets, transforms
from abc import abstractmethod


class iData(object):
    def __init__(self, root, name):
        self.root = root
        self.name = name
        self.template = ['a photo of a {}.']
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

    @abstractmethod
    def get_class_name(self):
        raise NotImplementedError()

    @abstractmethod
    def get_train_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self):
        raise NotImplementedError()


class iRSI(iData):
    def __init__(self, root, download=True):
        super(iRSI, self).__init__(root=root, name='iRSI')
        self.use_path = True

        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()

        self.gpt_prompt_path = 'descriptions/rsi_prompts.json'

        self.classes = ["airplane", "airport_runway", "artificial_grasland",
                        "avenue", "bare_land", "bridge", "city_building", "coastline", "container",
                        "crossroads", "dam", "desert", "dry_farm", "forest",
                        "green_farmland", "highway", "hirst", "lakeshore", "mangrove", "marina",
                        "mountain", "parkinglot", "pipeline", "residents", "river", "river_protection_forest",
                        "sandbeach", "sapling", "sea", "shrubwood", "snow_mountain", "sparse_forest",
                        "storage_room", "stream", "town"]

        self.download_data()

    def download_data(self):
        train_dir = os.path.join(self.root, "RSI-CB", "train")
        test_dir = os.path.join(self.root, "RSI-CB", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = self.split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = self.split_images_labels(test_dset.imgs)

    def get_class_name(self):
        return self.classes

    def get_train_data(self):
        return self.train_data, self.train_targets

    def get_test_data(self):
        return self.test_data, self.test_targets

    def split_images_labels(self, imgs):
        images = []
        labels = []
        for item in imgs:
            images.append(item[0])
            labels.append(item[1])

        return np.array(images), np.array(labels)


class iUCMerced(iData):
    def __init__(self, root, download=True):
        super(iUCMerced, self).__init__(root=root, name='iUCMerced')
        self.use_path = True

        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()

        self.gpt_prompt_path = 'descriptions/iUCMerced_prompts.json'

        self.classes = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings",
                        "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
                        "harbor", "intersection", "mediumresidential", "mobilehomepark", "overpass",
                        "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]

        self.download_data()

    def download_data(self):
        train_dir = os.path.join(self.root, "UCMerced_LandUse", "train")
        test_dir = os.path.join(self.root, "UCMerced_LandUse", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = self.split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = self.split_images_labels(test_dset.imgs)

    def get_class_name(self):
        return self.classes

    def get_train_data(self):
        return self.train_data, self.train_targets

    def get_test_data(self):
        return self.test_data, self.test_targets

    def split_images_labels(self, imgs):
        images = []
        labels = []
        for item in imgs:
            images.append(item[0])
            labels.append(item[1])

        return np.array(images), np.array(labels)


class iNWPU(iData):
    def __init__(self, root, download=True):
        super(iNWPU, self).__init__(root=root, name='iNWPU')
        self.use_path = True

        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()

        self.gpt_prompt_path = 'descriptions/nwpu_prompts.json'

        self.classes = ["airplane", "airport", "baseball_diamond", "basketball_court", "beach",
                        "bridge", "chaparral", "church", "circular_farmland", "cloud",
                        "commercial_area", "dense_residential", "desert", "forest", "freeway",
                        "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
                        "island", "lake", "meadow", "medium_residential", "mobile_home_park",
                        "mountain", "overpass", "palace", "parking_lot", "railway",
                        "railway_station", "rectangular_farmland", "river", "roundabout",
                        "runway", "sea_ice", "ship", "snowberg", "sparse_residential",
                        "stadium", "storage_tank", "tennis_court", "terrace", "thermal_power_station",
                        "wetland"]

        self.download_data()

    def download_data(self):
        train_dir = os.path.join(self.root, "NWPU-RESISC45", "train")
        test_dir = os.path.join(self.root, "NWPU-RESISC45", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = self.split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = self.split_images_labels(test_dset.imgs)

    def get_class_name(self):
        return self.classes

    def get_train_data(self):
        return self.train_data, self.train_targets

    def get_test_data(self):
        return self.test_data, self.test_targets

    def split_images_labels(self, imgs):
        images = []
        labels = []
        for item in imgs:
            images.append(item[0])
            labels.append(item[1])

        return np.array(images), np.array(labels)


class iAID(iData):
    def __init__(self, root, download=True):
        super(iAID, self).__init__(root=root, name='iAID')
        self.use_path = True

        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()

        self.gpt_prompt_path = 'descriptions/aid_prompts.json'

        self.classes = ["airport", "bareland", "baseballfield", "beach", "bridge",
                        "Center", "church", "commercial", "denseresidential",
                        "desert", "farmland", "forest", "industrial", "meadow",
                        "mediumresidential", "mountain", "park", "parking",
                        "playground", "pond", "port", "railwaystation", "resort",
                        "river", "school", "sparseresidential", "square", "stadium",
                        "storagetanks", "viaduct"]

        self.download_data()

    def download_data(self):
        train_dir = os.path.join(self.root, "AID", "train")
        test_dir = os.path.join(self.root, "AID", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = self.split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = self.split_images_labels(test_dset.imgs)

    def get_class_name(self):
        return self.classes

    def get_train_data(self):
        return self.train_data, self.train_targets

    def get_test_data(self):
        return self.test_data, self.test_targets

    def split_images_labels(self, imgs):
        images = []
        labels = []
        for item in imgs:
            images.append(item[0])
            labels.append(item[1])

        return np.array(images), np.array(labels)


def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    return t
