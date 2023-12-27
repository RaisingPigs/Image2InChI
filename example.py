import pandas as pd

from image2inchi.model import Image2InChI, test_public_dataset
from image2inchi.utils import Scorer

model = Image2InChI(chk_pt='/weight/chk_pt.pth')

inchi = model.pred_one('./image2inchi/data/dataset/BMS1000/images/000029a61c01.png')
print(inchi)
