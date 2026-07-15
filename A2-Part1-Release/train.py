from basic_fcn import *
from experimental_fcn import TransferFCN
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import multiprocessing
import argparse
import os
import matplotlib.pyplot as plt

# a handful of persistent workers beats one worker pool respawn per epoch
# (macOS uses spawn, which re-imports torch in every new worker)
num_workers = min(4, multiprocessing.cpu_count())

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases


def getClassWeights():
    """Inverse-frequency class weights over the training masks (for Q4.c-style
    weighted loss). Rare classes get proportionally larger weights."""
    counts = torch.zeros(n_class, dtype=torch.float64)
    for _, labels in train_loader:
        counts += torch.bincount(labels.reshape(-1), minlength=n_class).double()
    freq = counts / counts.sum()
    weights = 1.0 / torch.sqrt(freq + 1e-8)
    weights = weights / weights.sum() * n_class  # normalize to mean 1
    return weights.float()


def train():
    """
    Train a deep learning model using mini-batches.

    - Perform forward propagation in each epoch.
    - Compute loss and conduct backpropagation.
    - Update model weights.
    - Evaluate model on validation set for mIoU score.
    - Implement early stopping if necessary.

    Returns:
        None.
    """

    best_iou_score = 0.0
    patience = 8
    epochs_without_improvement = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        ts = time.time()
        epoch_losses = []
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model(inputs)  # (N, n_class, H, W)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            epoch_losses.append(loss.item())
            if iter % 20 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        if scheduler is not None:
            scheduler.step()

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        current_miou_score, current_val_loss = val(epoch)
        train_losses.append(np.mean(epoch_losses))
        val_losses.append(current_val_loss)

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            epochs_without_improvement = 0
            # save the best model seen so far
            if not os.path.exists(models_loc):
                os.makedirs(models_loc)
            torch.save(fcn_model.state_dict(), model_path)
            print(f"Saved new best model (IoU {best_iou_score:.4f}) to {model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    plot_losses(train_losses, val_losses, args.experiment)


def plot_losses(train_losses, val_losses, fname):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss per Epoch ({fname})')
    plt.legend()
    plt.savefig(f"./plots/{fname}.png")
    plt.close()


def val(epoch):
    """
    Validate the deep learning model on a validation dataset.

    Returns:
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            pred = outputs.argmax(dim=1)
            mean_iou_scores.append(util.iou(pred, labels, n_class))
            accuracy.append(util.pixel_acc(pred, labels))

    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(losses)


def modelTest():
    """
    Test the deep learning model using a test dataset.

    Returns:
        None. Outputs average test metrics to the console.
    """

    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    total_correct = 0
    total_pixels = 0
    total_intersection = np.zeros(n_class, dtype=np.int64)
    total_union = np.zeros(n_class, dtype=np.int64)

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model(inputs)
            pred = outputs.argmax(dim=1)

            total_correct += (pred == labels).sum().item()
            total_pixels += labels.numel()
            intersection, union = util.iou_components(pred, labels, n_class)
            total_intersection += intersection
            total_union += union

    final_avg_pixel_acc = total_correct / total_pixels
    # IoU per class over the entire test set, then averaged over classes
    present = total_union > 0
    final_avg_iou = float(np.mean(total_intersection[present] / total_union[present]))

    #print the final results using the following output for autograding purposes
    print(f"Final average pixel accuracy: {final_avg_pixel_acc:.4f}, final average IoU: {final_avg_iou:.4f}")

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def exportModel(inputs):
    """
    Export the output of the model for given inputs.

    Args:
        inputs: Input data to the model.

    Returns:
        Output from the model for the given inputs.
    """

    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !

    saved_model_path = model_path
    fcn_model.load_state_dict(torch.load(saved_model_path, map_location=device))

    inputs = inputs.to(device)

    output_image = fcn_model(inputs)

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return output_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', dest="experiment", type=str, default='baseline', help='Specify the experiment that you want to run')
    parser.add_argument('--data_dir', type=str, default='./data', help= 'Specify the directory that your VOC data is located')
    args = parser.parse_args()

    # normalize using imagenet averages
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

    target_transform = MaskToTensor()

    train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform, data_root=args.data_dir)
    val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform, data_root=args.data_dir)
    test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform, data_root=args.data_dir)

    train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False, num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False, num_workers=num_workers, persistent_workers=num_workers > 0)

    epochs = 60

    n_class = 21

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    models_loc = "./models"
    model_path = f"{models_loc}/{args.experiment}.pth"

    scheduler = None

    if args.experiment == "baseline":
        fcn_model = FCN(n_class=n_class) #this should match the model architecture of the .pth file
        # AdamW with weight decay: the FCN overfits the 1464 train images
        # quickly (train loss < 0.2 by epoch 30), so regularization is needed
        optimizer = torch.optim.AdamW(fcn_model.parameters(), lr=1e-3, weight_decay=1e-3)
        # the output layer produces per-pixel class scores (softmax), so the
        # matching criterion is per-pixel cross entropy
        criterion = nn.CrossEntropyLoss()

    elif args.experiment == "improved_baseline":
        # Q4 (a): baseline + cosine annealing learning rate schedule
        fcn_model = FCN(n_class=n_class)
        optimizer = torch.optim.Adam(fcn_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    elif args.experiment == "experimental":
        # Q5 (b): transfer learning, ImageNet-pretrained ResNet34 encoder
        fcn_model = TransferFCN(n_class=n_class)
        optimizer = torch.optim.Adam([
            {'params': fcn_model.encoder.parameters(), 'lr': 1e-4},  # fine-tune gently
            {'params': [p for n, p in fcn_model.named_parameters() if not n.startswith('encoder')], 'lr': 1e-3},
        ])
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    if os.path.exists(model_path): #model exists, load it
        state_dict = torch.load(model_path, map_location=device)
        fcn_model.load_state_dict(state_dict)
        fcn_model = fcn_model.to(device)
        modelTest()

    else: #model does not exist, train and save it
        if args.experiment == "experimental":
            # keep the pretrained encoder weights; initialize only the decoder
            for name, module in fcn_model.named_modules():
                if not name.startswith('encoder'):
                    init_weights(module)
        else:
            fcn_model.apply(init_weights)
        fcn_model = fcn_model.to(device)
        val(0)  # show the accuracy before training
        train()
        # reload the best checkpoint saved during training and report test metrics
        fcn_model.load_state_dict(torch.load(model_path, map_location=device))
        modelTest()

    # housekeeping
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
