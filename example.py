import pandas as pd

from image2inchi.model import Image2InChI, test_public_dataset
from image2inchi.utils import Scorer

model = Image2InChI(chk_pt='/weight/chk_pt.pth')
# 
inchi = model.pred_one('./image2inchi/data/dataset/BMS1000/images/000029a61c01.png')
print(inchi)

# test_public_dataset(model)  
# 
# inchi_df = pd.read_csv('/usr/local/mine/python/miniconda-project/swint_t_pkg/data/dataset/UOB/UOB_inchi.csv')
# pred_df = pd.read_csv('/usr/local/mine/python/miniconda-project/swint_t_pred/log/uob_result.csv')
# 
# pred_dict = dict()
# for index, row in pred_df.iterrows():
#     pred_dict[str(row['img_name'])] = row['InChI_pred']
# 
# inchi_list = []
# pred_list = []
# for index, row in inchi_df.iterrows():
#     inchi_list.append(row['InChI'])
#     pred_list.append(pred_dict[str(row['img_name'])])
# 
# inchi_acc, morgan_fp, mcs, lcs = Scorer.scoring(inchi_list, pred_list)
# 
# print(inchi_acc, morgan_fp, mcs, lcs)
