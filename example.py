from image2inchi.model import Image2InChI

model = Image2InChI(chk_pt='/weight/chk_pt.pth')

inchi = model.pred_one('./image2inchi/data/dataset/BMS1000/images/000029a61c01.png')
print(inchi)
