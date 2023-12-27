import os
import torch
from pathlib import Path


class PathConfig:
    # 基本文件路径
    BASE_DIR = os.path.dirname(Path(__file__).resolve())
    DATA_DIR = BASE_DIR + '/data'
    DATASET_DIR = DATA_DIR + '/dataset'

    # 原始数据集文件路径
    BMS1000_DIR = DATASET_DIR + '/BMS1000'
    BMS1000_IMAGES_DIR = BMS1000_DIR + '/images'

    CLEF_DIR = DATASET_DIR + '/CLEF'
    CLEF_IMAGES_DIR = CLEF_DIR + '/images'

    JPO_DIR = DATASET_DIR + '/JPO'
    JPO_IMAGES_DIR = JPO_DIR + '/images'

    UOB_DIR = DATASET_DIR + '/UOB'
    UOB_IMAGES_DIR = UOB_DIR + '/images'

    USPTO_DIR = DATASET_DIR + '/USPTO'
    USPTO_IMAGES_DIR = USPTO_DIR + '/images'

    # 数据预处理得到的文件路径
    BMS1000_INCHI_CSV = BMS1000_DIR + '/BMS1000_inchi.csv'
    CLEF_INCHI_CSV = CLEF_DIR + '/CLEF_inchi.csv'
    JPO_INCHI_CSV = JPO_DIR + '/JPO_inchi.csv'
    UOB_INCHI_CSV = UOB_DIR + '/UOB_inchi.csv'
    USPTO_INCHI_CSV = USPTO_DIR + '/USPTO_inchi.csv'

    TOKEN_STOI_PICKLE = DATA_DIR + '/tokenizer.stoi.pickle'
    LOAD_WEIGHT_PATH = DATA_DIR + '/chk_pt.pth'


# 训练时常量
class TrainConfig:
    # 模型参数
    ENCODER_DIM = 512
    EMBED_DIM = 512
    N_HEAD = 8
    FF_DIM = 1024
    NUM_LAYER = 3
    SIZE = 224

    N_FOLD = 5
    SEED = 42

    BATCH_SIZE = 32
    NUM_WORKERS = 2
    EPOCHS = 40
    LR = 1e-4
    SCHEDULER_NAME = 'CosineAnnealingWarmRestarts'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    DROPOUT = 0.5
    WEIGHT_DECAY = 1e-6
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PRINT_FREQ = 10
    MAX_LEN = 300

    START_EPOCH = 40


class Config:
    PATH = PathConfig
    TRAIN = TrainConfig
