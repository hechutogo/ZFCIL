from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
import argparse
from models.engine import Runner
import copy


def extend_cfg(cfg):
    # Device setting
    cfg.DEVICE = CN()
    cfg.DEVICE.DEVICE_NAME = ''
    cfg.DEVICE.GPU_ID = ''

    cfg.SEED = []

    # For dataset config
    cfg.DATASET = CN()
    cfg.DATASET.NAME = ''
    cfg.DATASET.ROOT = ''
    cfg.DATASET.GPT_PATH = ''
    cfg.DATASET.M = -1
    cfg.DATASET.NUM_INIT_CLS = -1
    cfg.DATASET.NUM_INC_CLS = -1
    cfg.DATASET.NUM_BASE_SHOT = -1
    cfg.DATASET.NUM_INC_SHOT = -1
    cfg.DATASET.BETA = -1.0
    cfg.DATASET.SHUFFLE = False

    # For data
    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE = -1
    cfg.DATALOADER.TRAIN.BATCH_SIZE_INC = -1
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = -1
    cfg.DATALOADER.NUM_WORKERS = -1

    # For model
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ''

    # For methods
    cfg.TRAINER = CN()
    cfg.TRAINER.ZFCIL = CN()
    cfg.TRAINER.ZFCIL.PREC = ''
    cfg.TRAINER.ZFCIL.VISION_CALIBRATION = False


def setup_cfg(config_file):
    cfg = CN()
    extend_cfg(cfg)
    cfg.merge_from_file(config_file)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    cfg = setup_cfg(args.config)

    seed_list = copy.deepcopy(cfg.SEED)
    for seed in seed_list:
        cfg.SEED = seed
        set_seed(cfg.SEED)
        set_gpu(cfg.DEVICE.GPU_ID)
        engine = Runner(cfg)
        engine.run()


if __name__ == '__main__':
    main()
