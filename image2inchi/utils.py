import math
import pickle
import time
import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
import Levenshtein
# albumentations数据增强库
from albumentations.pytorch import ToTensorV2
import albumentations
from datetime import datetime
from image2inchi.config import Config
import os
import random
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit import DataStructs
from rdkit.Chem import AllChem
import requests
from tqdm.auto import tqdm
import urllib3

tqdm.pandas()


def mkdir(dir_path):
    is_exists = os.path.exists(dir_path)

    if not is_exists:
        os.makedirs(dir_path)


def seed_torch(seed=42):
    """
    主要用来设置各种包的随机种子, 将所有包的随机种子固定为一个值, 可以使得结果可复现
    :param seed: 
    """
    random.seed(seed)
    # os.environ 获取环境变量
    # PYTHONHASHSEED python的hash种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # deterministic置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    torch.backends.cudnn.deterministic = True


class Tokenizer():
    def __init__(self, filepath=None):
        """初始化方法
        生成 stoi 字典和 itos 字典
        stoi : {char:int} 字符和index映射
        itos : {int:char} index和字符映射
        """
        self.stoi = {}
        self.itos = {}

        if filepath:
            with open(filepath, 'rb') as f:
                self.stoi = pickle.load(f)
            self.itos = {k: v for v, k in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def create_dicts_for_texts(self, texts):
        """根据文本生成字典
        :param: list, text 为分词后形成的 list e.g ['C 13 H 20 O S','C 21 H 30 O 4',...]
        """
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_seq(self, text):
        """将 text 转换成 int list, 加头<sos>尾<eos>
            输入text='C 13 H 20 O S', 返回sequence=[190,98,0,23,4,54,43,191]
        """
        try:
            sequence = []
            sequence.append(self.stoi['<sos>'])
            for s in text.split(' '):
                sequence.append(self.stoi[s])
            sequence.append(self.stoi['<eos>'])
            return sequence
        except:
            return []

    def texts_to_seqs(self, texts):
        """将多个 text 转换成 intlist
        """
        sequences = []
        for text in texts:
            sequence = self.text_to_seq(text)
            sequences.append(sequence)
        return sequences

    def seq_to_text(self, sequence):
        """将 intlist 转换成 text
            输入sequence=[190,98,0,23,4,54,43,191], 返回text='C 13 H 20 O S'
        """
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def seqs_to_texts(self, sequences):
        """将多个 intlist 转换成text
        """
        texts = []
        for sequence in sequences:
            text = self.seq_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        """将预测结果 (intlist) 转换为字符 (str)，组装为标准 InChI 格式
        e.g [190, 178, 47, 182, 89, 185, 187, 6, 13, 4, 165, 0, 88, 1, 154, 4, 69, 4, 47, 4, 132, 4, 121, 4, 14, 0, 99, 1, 143, 4, 36, 0, 47, 1, 25, 0, 110, 1, 58, 7, 121, 4, 143, 3, 165, 3, 25, 3, 58, 182, 3, 154, 182, 88, 3, 13, 4, 110, 182, 99, 191]
            ->
            InChI=1S/C13H20OS/c1-9(2)8-15-13-6-5-10(3)7-12(13)11(4)14/h5-7,9,11,14H,8H2,1-4H3 
        """
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        """将多个预测结果 (intlist) 转换为字符 (text)，组装为标准 InChI 格
        """
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions

    def get_seq_of_sos(self):
        return self.stoi['<sos>']

    def get_seq_of_eos(self):
        return self.stoi['<eos>']

    def get_seq_of_pad(self):
        return self.stoi['<pad>']


class TrainDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, filepath: str, transform):
        super().__init__()
        self.data_df = data_df
        self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img_path = self.data_df['img_path'][index]
        img_path = self.filepath + img_path
        # 这里是以三通道的方式去读img的, 所以image直接就是三通道
        image = cv2.imread(img_path)

        # 将BGR格式转换成RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        augmented = self.transform(image=image)
        image_tsr = augmented['image']

        label = self.data_df['seq'][index]
        label_len = self.data_df['seq_len'][index]

        # transform已经将image变为tensor了
        return image_tsr, \
               torch.tensor(label).long(), \
               torch.tensor(label_len).long(), \
               self.data_df['InChI'][index]


class TestDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, filepath: str, transform):
        super().__init__()
        self.data_df = data_df
        self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img_path = self.data_df['img_path'][index]
        img_path = self.filepath + img_path
        # 这里是以三通道的方式去读img的, 所以image直接就是三通道
        image = cv2.imread(img_path)

        # 将BGR格式转换成RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        augmented = self.transform(image=image)
        image_tsr = augmented['image']

        inchi_text = self.data_df['InChI'][index]

        # transform已经将image变为tensor了
        return image_tsr, inchi_text


def get_transforms():
    return albumentations.Compose([
        albumentations.Resize(Config.TRAIN.SIZE, Config.TRAIN.SIZE),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_logger(log_filepath):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_filepath)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    with open(file=log_filepath, mode="a") as f:
        f.seek(0)
        f.truncate()

    return logger


class Scorer:
    @staticmethod
    def scoring(label_texts, label_text_preds):
        inchi_acc = Scorer.get_inchi_acc(label_texts, label_text_preds)
        morgan_fp = Scorer.get_morgan_fp(label_texts, label_text_preds)
        mcs = Scorer.get_mcs_acc(label_texts, label_text_preds)
        # distance = Scorer.get_distance(label_texts, label_text_preds)
        lcs = Scorer.get_lcs(label_texts, label_text_preds)

        return inchi_acc, morgan_fp, mcs, lcs

    @staticmethod
    def get_distance(y_true, y_pred):
        scores = []
        for true, pred in zip(y_true, y_pred):
            score = Levenshtein.distance(true, pred)
            scores.append(score)
        avg_score = np.mean(scores)
        return avg_score

    @staticmethod
    def get_lcs(y_true, y_pred):
        lcss = []
        for true, pred in tqdm(zip(y_true, y_pred), desc='lcs_acc'):
            lcs = Scorer.longest_common_subsequence(true, pred)
            lcss.append(lcs / len(true))
        avg_lcs = np.mean(lcss)
        return avg_lcs

    @staticmethod
    def longest_common_subsequence(text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    @staticmethod
    def get_single_mcs_acc(inchi, inchi_pred):
        mol = Chem.inchi.MolFromInchi(inchi)
        mol_pred = Chem.inchi.MolFromInchi(inchi_pred)

        if mol is None or mol_pred is None:
            return 0

        mcs_res = rdFMCS.FindMCS([mol, mol_pred], timeout=30)

        mol_atom_num = mol.GetNumAtoms()  # 原子数
        mol_bond_num = len(mol.GetBonds())  # 连接数

        mcs_atom_num = mcs_res.numAtoms
        mcs_bond_num = mcs_res.numBonds

        return (mcs_atom_num + mcs_bond_num) / (mol_atom_num + mol_bond_num)

    @staticmethod
    def get_mcs_acc(label_texts, label_text_preds):
        total = 0
        total_mcs_acc = 0

        for label_text, label_text_pred in tqdm(zip(label_texts, label_text_preds), desc='mcs_acc'):
            total_mcs_acc += Scorer.get_single_mcs_acc(label_text, label_text_pred)
            total += 1

        return total_mcs_acc / total

    @staticmethod
    def get_inchi_acc(label_texts, label_text_preds):
        total = 0
        right = 0

        for label_text, label_text_pred in tqdm(zip(label_texts, label_text_preds), desc='inchi_acc'):
            total += 1
            try:
                res = Chem.inchi.MolFromInchi(label_text_pred)
                if res is not None:
                    right += 1
            except Exception as e:
                pass

        return right / total

    @staticmethod
    def get_single_morgan_fp(inchi, inchi_pred):
        mol = Chem.MolFromInchi(inchi)
        mol_pred = Chem.MolFromInchi(inchi_pred)

        if mol is None or mol_pred is None:
            return 0

        fp = AllChem.GetMorganFingerprint(mol, 2)
        fp_pred = AllChem.GetMorganFingerprint(mol_pred, 2)

        return DataStructs.DiceSimilarity(fp, fp_pred)

    @staticmethod
    def get_morgan_fp(label_texts, label_text_preds):
        total = 0
        total_morgan_fp_acc = 0

        for label_text, label_text_pred in tqdm(zip(label_texts, label_text_preds), desc='morgan_fp'):
            total_morgan_fp_acc += Scorer.get_single_morgan_fp(label_text, label_text_pred)
            total += 1

        return total_morgan_fp_acc / total


class AverageMeter:
    """记录总数和平均数的类"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def seq_loss_calculate(loss_fn, seq_preds, seq_truth, seq_lens):
    # 要去掉真实token的第一个sos
    seq_preds = pack_padded_sequence(seq_preds, seq_lens, batch_first=True).data
    seq_truth = pack_padded_sequence(seq_truth, seq_lens, batch_first=True).data

    return loss_fn(seq_preds, seq_truth)


def get_now_str():
    return datetime.now().strftime('%Y_%m_%d')


def is_inchi(inchi):
    res = Chem.MolFromInchi(inchi)
    return res is not None


class InChIQuery:
    @staticmethod
    def query(inchi):
        if is_inchi(inchi):
            return inchi

        inchi_formula = inchi.split('/')[1]
        queried_inchi_list = InChIQuery.query_inchi_list(inchi_formula)
        if not queried_inchi_list:
            return inchi

        queried_inchi = InChIQuery.get_most_matching_inchi(inchi, queried_inchi_list)

        if queried_inchi and is_inchi(queried_inchi):
            return queried_inchi

        return inchi

    @staticmethod
    def req_get(url, param):
        # 禁用 InsecureRequestWarning 警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # 发起请求时忽略证书验证警告
        requests.packages.urllib3.disable_warnings()

        resp = requests.get(url, params=param, verify=False)
        return resp.json()

    @staticmethod
    def query_cache_key(inchi_formula):
        url = 'https://pubchem.ncbi.nlm.nih.gov/unified_search/structure_search.cgi'
        param = {
            'format': 'json',
            'queryblob': '{"query":{"type":"formula","parameter":[{"name":"FormulaQuery","string":"' + inchi_formula + '"},{"name":"UseCache","bool":true},{"name":"SearchTimeMsec","num":5000},{"name":"SearchMaxRecords","num":100000},{"name":"allowotherelements","bool":false}]}}'
        }

        res = InChIQuery.req_get(url, param)
        resp = res['response']

        assert resp['status'] == 0, 'status!=0, 请求异常'
        assert resp['cachekey'], 'cachekey不存在'
        return resp['cachekey']

    @staticmethod
    def query_inchi_list_by_cache_key(cache_key):
        url = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi'
        param = {
            'infmt': 'json',
            'outfmt': 'json',
            'query': '{"select":"*","collection":"compound","where":{"ands":[{"input":{"type":"netcachekey","idtype":"cid","key":"' + cache_key + '"}}]},"order":["relevancescore,desc"],"start":1,"limit":100,"width":1000000,"listids":0}'
        }

        res = InChIQuery.req_get(url, param)
        resp = res['SDQOutputSet'][0]

        assert resp['status']['code'] == 0, 'status!=0, 请求异常'

        rows = resp['rows']

        inchi_list = []
        for row in rows:
            inchi_dict = {
                'inchi': row['inchi'],
                'inchi_formula': row['mf'],
                'inchi_key': row['inchikey']
            }
            inchi_list.append(inchi_dict)

        return inchi_list

    @staticmethod
    def query_inchi_list(inchi_formula):
        try:
            cache_key = InChIQuery.query_cache_key(inchi_formula)
            return InChIQuery.query_inchi_list_by_cache_key(cache_key)
        except Exception as e:
            return None

    @staticmethod
    def get_most_matching_inchi(inchi, inchi_list):
        max_score = 0
        queried_inchi_of_max_score = ''

        for inchi_dict in inchi_list:
            queried_inchi = inchi_dict['inchi']
            diff_score = Levenshtein.ratio(inchi, queried_inchi)
            if diff_score > max_score:
                queried_inchi_of_max_score = queried_inchi
                max_score = diff_score

        return queried_inchi_of_max_score
