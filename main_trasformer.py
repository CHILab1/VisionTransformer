import os
import torch
import shutil
import pandas as pd
import torch.nn as nn
from training import *
import torch.optim as optim
import torchvision.transforms as T
from vit_personalizzato import SimpleViT
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":

    # Defingn the paths
    rootdir = ""
    
    seed=42
    set_seed(seed)
    
    numero_prova=1
    datadir = os.path.join(rootdir, "root_path")
    savedir = os.path.join(rootdir, f"dst_ditr/prova_{numero_prova}")
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, "models"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "results"), exist_ok=True)
    # csv = pd.read_csv(os.path.join(datadir, "all_classes.csv"))
    csv = pd.read_csv(os.path.join(datadir, "merged_complete_1.csv"))
    
    train_csv, validation_csv = train_test_split(csv, test_size=0.2, random_state=seed)

    train_csv= train_csv.reset_index(drop=True)
    validation_csv= validation_csv.reset_index(drop=True)
    
    transforms = T.Compose(
        [T.ToTensor(), T.Resize((224, 224), antialias=True)])
    train = Dataset_v2(root=True, csv=train_csv, transform=transforms, mask=True)
    val = Dataset_v2(root=True, csv=validation_csv, transform=transforms, mask=True)
    train_loader = DataLoader(train, batch_size=64,
                            shuffle=True, drop_last=True, num_workers=16)
    val_loader = DataLoader(val, batch_size=64, shuffle=True, drop_last=True, num_workers=16)

    # Defining the model and hyperparameters
    lr = 0.0001
    epochs = 5000
    max_patience = 50
    inchannels =3
    classes=8
    device= torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = SimpleViT(
        image_size = 224,
        patch_size = 32,
        num_classes = 8,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        output_class_dims=[6, 6, 3, 3, 6, 6, 6, 6]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    _,_ = training(iterations=0, model=model, path=savedir, lr=lr, best_loss=9999, last_epoch=0, epochs=epochs, train_loader=train_loader, test_loader=val_loader,
            val_loader=val_loader, device=device, patience_max=max_patience, criterion=criterion, nclasses=8, optimizer=optimizer, type="classificazione")

