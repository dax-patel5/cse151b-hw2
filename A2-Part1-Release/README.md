# PA2 - Semantic Segmentation using FCN (A1 Part 2 + A2 Part 1)

## Setup

Python 3 with PyTorch, torchvision, numpy, matplotlib, and Pillow is required
(the provided `environment.yml` works: `conda env create -f environment.yml`).

Download the PASCAL VOC-2012 dataset first (creates the `data/` folder):

```bash
python download.py
```

## Files

- `voc.py` — VOC dataset class. Train split = `train.txt`, val split = `trainval.txt`,
  test split = `val.txt` (as specified in the starter comments). Boundary pixels
  (255) are mapped to background (0).
- `util.py` — `pixel_acc` and `iou` metrics, plus `iou_components` which returns
  per-class intersection/union counts so test-set IoU can be accumulated over the
  whole set before averaging over classes.
- `basic_fcn.py` — baseline encoder/decoder FCN. Five stride-2 convolutions
  (3→32→64→128→256→512) and five stride-2 transposed convolutions
  (512→512→256→128→64→32) followed by a 1x1 classifier, so a 224x224 input
  produces 224x224 per-pixel class scores.
- `experimental_fcn.py` — `TransferFCN`: ImageNet-pretrained ResNet34 encoder
  (avgpool + fc removed) with the same five-deconv decoder (Q5 option b).
- `train.py` — training / validation / test loops with early stopping
  (patience 5 on validation mean IoU; the best checkpoint is saved).
- `pixel.txt` — RGB value of the 7th pixel (index 6) of the 100th image
  (index 99) of the test set, after the 224x224 resize and `ToTensor`
  (values in [0, 1], before normalization).

## Running

Each experiment trains (30 epochs max, Adam, cross-entropy loss, early stopping)
if `models/<experiment>.pth` does not exist, then evaluates on the test set. If
the saved model already exists it is loaded and only evaluated:

```bash
python train.py --experiment baseline            # basic FCN
python train.py --experiment improved_baseline   # + cosine annealing LR schedule (Q4 a)
python train.py --experiment experimental        # ResNet34 transfer-learning encoder (Q5 b)
```

Each run prints per-epoch train loss and validation loss / pixel accuracy / mean
IoU, saves loss curves to `plots/<experiment>.png`, saves the best model to
`models/<experiment>.pth`, and finishes with the autograded line:

```
Final average pixel accuracy: xxxx, final average IoU: xxxx
```

Test-set pixel accuracy is computed globally (total correct / total pixels) and
test-set IoU is computed per class over the entire test set, then averaged over
the classes that appear.

`--data_dir` can point at a different VOC data root if needed (default `./data`).
