# Image2InChI

## descirption

This is the repository for Image2InChI, an image-to-graph model that translates a molecular image to its InChI.

## Quick Start

### Installation

Install Image2InChI with pip. 

Note that Image2InChI supports Linux systems, but due to related dependencies and other
reasons, it currently does not support Windows

```shell
pip install Image2InChI
```

Download the Image2InChI checkpoint (chk_pt.pth) from HuggingFace
Hub: https://huggingface.co/RaisingPigs/Image2InChI/tree/main

### Example

Predict based on a picture

```python
from image2inchi.model import Image2InChI

model = Image2InChI(chk_pt='/weight/chk_pt.pth')

inchi_pred = model.pred_one('/data/aaa.png')
print(inchi_pred)
```

Using the pred_batch(), pass in a folder path to predict This function runs the Image2InChI model on all images in a
given directory and saves the results in `results.csv` in the same directory. Each line of the output file contains the
image file name and the InChI representation of the described molecule.

```python
from image2inchi.model import Image2InChI

model = Image2InChI(chk_pt='/weight/chk_pt.pth')

inchi_df = model.pred_batch('/data/bms_test')
print(inchi_df)
```

### Public Dataset

Validate using a publicly available dataset

```python
from image2inchi.model import Image2InChI, test_bms1000, test_jpo, test_clef, test_uob, test_uspto

model = Image2InChI(chk_pt='/weight/chk_pt.pth')

test_bms1000(model)
test_jpo(model)
test_clef(model)
test_uob(model)
test_uspto(model)
```

You can use the test_public_dataset() to verify the above

```python
from image2inchi.model import Image2InChI, test_public_dataset

model = Image2InChI(chk_pt='/weight/chk_pt.pth')
test_public_dataset(model)
```

### Evaluation

scoring metrics can be obtained using the Scorer.scoring(). The function needs to pass in two iterable parameters,
parameter 1 is the InChI label and parameter 2 is the predicted InChI.

```python
from image2inchi.utils import Scorer

inchi_list = []  # InChI label list
inchi_pred_list = []  # predicted InChI list
inchi_acc, morgan_fp, mcs, lcs = Scorer.scoring(inchi_list, inchi_pred_list)
print(f'result: inchi_acc - {inchi_acc}, morgan_fp - {morgan_fp}, mcs - {mcs}, lcs - {lcs}')
```