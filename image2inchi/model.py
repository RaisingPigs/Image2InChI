import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from image2inchi.config import Config
from image2inchi.utils import Tokenizer, get_transforms, seed_torch, InChIQuery, Scorer
from image2inchi.model_core import InCHImgAnalyzer
from rdkit import RDLogger

tqdm.pandas()

IMGTYPE_LIST = {'.jpg', '.bmp', '.png', '.jpeg', '.jfif'}


class Image2InChI:
    def __init__(self, chk_pt):
        RDLogger.DisableLog('rdApp.*')
        seed_torch(Config.TRAIN.SEED)
        self.chk_pt = chk_pt
        self.tokenizer = Tokenizer(Config.PATH.TOKEN_STOI_PICKLE)
        self.transform = get_transforms()
        self.net = InCHImgAnalyzer(
            encoder_dim=Config.TRAIN.ENCODER_DIM,
            vocab_size=len(self.tokenizer),
            embed_dim=Config.TRAIN.EMBED_DIM,
            max_length=Config.TRAIN.MAX_LEN,
            num_head=Config.TRAIN.N_HEAD,
            ff_dim=Config.TRAIN.FF_DIM,
            num_layer=Config.TRAIN.NUM_LAYER
        )

        self.init_net()

    def pred_batch(self, img_dir: str) -> pd.DataFrame:
        df = pd.DataFrame(columns=['img_name', 'InChI_pred'])

        for img_name in tqdm(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, img_name)

            if os.path.isdir(img_path):
                print(f'{img_name}, This file is not a picture, skip')
                continue

            _, postfix = os.path.splitext(img_name)
            if postfix not in IMGTYPE_LIST:
                print(f'{img_name}, This file is not a picture, skip')
                continue

            inchi = self.pred_one(img_path)
            df.loc[len(df)] = [img_name, inchi]

        df.to_csv(os.path.join(img_dir, 'results.csv'))

        return df

    def pred_one(self, img_path: str):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        augmented = self.transform(image=image)
        image_tsr = augmented['image']
        image_tsr = image_tsr.unsqueeze(dim=0)

        return self.do_test(image_tsr)

    def init_net(self):
        self.net.to(Config.TRAIN.DEVICE)
        states = torch.load(self.chk_pt, map_location=torch.device(Config.TRAIN.DEVICE))
        self.net.load_state_dict(states['net'], strict=False)
        print(f'Loading weight...')

    def do_test(self, img):
        self.net.eval()
        img = img.to(Config.TRAIN.DEVICE)

        with torch.no_grad():
            preds = self.net.predict(
                img,
                Config.TRAIN.MAX_LEN,
                self.tokenizer.get_seq_of_sos(),
                self.tokenizer.get_seq_of_eos(),
                self.tokenizer.get_seq_of_pad()
            )

        label_text_pred = self.tokenizer.predict_captions(preds.detach().cpu().numpy())
        return InChIQuery.query(f'InChI=1S/{label_text_pred[0]}')


def test_dataset(model: Image2InChI, img_dir: str, inchi_csv_path: str):
    inchi_df = pd.read_csv(inchi_csv_path)
    pred_df = model.pred_batch(img_dir)

    pred_dict = dict()
    for index, row in pred_df.iterrows():
        pred_dict[str(row['img_name'])] = row['InChI_pred']

    inchi_list = []
    pred_list = []
    for index, row in inchi_df.iterrows():
        inchi_list.append(row['InChI'])
        pred_list.append(pred_dict[str(row['img_name'])])

    inchi_acc, morgan_fp, mcs, lcs = Scorer.scoring(inchi_list, pred_list)

    return inchi_acc, morgan_fp, mcs, lcs


def test_bms1000(model: Image2InChI):
    print('bms1000')
    inchi_acc, morgan_fp, mcs, lcs = test_dataset(model, Config.PATH.BMS1000_IMAGES_DIR, Config.PATH.BMS1000_INCHI_CSV)
    print(f'bms1000 result: inchi_acc - {inchi_acc}, morgan_fp - {morgan_fp}, mcs - {mcs}, lcs - {lcs}')


def test_jpo(model: Image2InChI):
    print('jpo')
    inchi_acc, morgan_fp, mcs, lcs = test_dataset(model, Config.PATH.JPO_IMAGES_DIR, Config.PATH.JPO_INCHI_CSV)
    print(f'jpo result: inchi_acc - {inchi_acc}, morgan_fp - {morgan_fp}, mcs - {mcs}, lcs - {lcs}')


def test_clef(model: Image2InChI):
    print('clef')
    inchi_acc, morgan_fp, mcs, lcs = test_dataset(model, Config.PATH.CLEF_IMAGES_DIR, Config.PATH.CLEF_INCHI_CSV)
    print(f'clef result: inchi_acc - {inchi_acc}, morgan_fp - {morgan_fp}, mcs - {mcs}, lcs - {lcs}')


def test_uob(model: Image2InChI):
    print('uob')
    inchi_acc, morgan_fp, mcs, lcs = test_dataset(model, Config.PATH.UOB_IMAGES_DIR, Config.PATH.UOB_INCHI_CSV)
    print(f'uob result: inchi_acc - {inchi_acc}, morgan_fp - {morgan_fp}, mcs - {mcs}, lcs - {lcs}')


def test_uspto(model: Image2InChI):
    print('uspto')
    inchi_acc, morgan_fp, mcs, lcs = test_dataset(model, Config.PATH.USPTO_IMAGES_DIR, Config.PATH.USPTO_INCHI_CSV)
    print(f'uspto result: inchi_acc - {inchi_acc}, morgan_fp - {morgan_fp}, mcs - {mcs}, lcs - {lcs}')


def test_public_dataset(model: Image2InChI):
    test_bms1000(model)
    test_jpo(model)
    test_clef(model)
    test_uob(model)
    test_uspto(model)
