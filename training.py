import os
import sys
import cv2
import torch
import torch.utils
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset 
from sklearn.metrics import accuracy_score, classification_report


def train_new(model, lr, epoch, train_loader,device, patience, criterion, optimizer=None, type='classificazione', iteration=0, train=True):
    optimizer = optimizer
    cross_entropy_list = []
    e_sum = [[],[],[]]
    model.train()
    mode= "Train" if train else "Val"
    for idx, info in tqdm(enumerate(train_loader), total=len(train_loader)):
        image, label = info
        image = image.to(device)
        label = label.to(device)
        preds = model(image) 
        if train:
            optimizer.zero_grad()
        if type== 'classificazione':
            loss = criterion(preds.squeeze(1), label)
            e_sum[0].append(loss.item())
        cross_entropy_list.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()
        image = image.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        del image
        del label
        del preds
    curr_lr = 0
    if type== 'classificazione':
        print(f"Epoch: {epoch} Avg_{mode}_Loss: {sum(cross_entropy_list)/len(cross_entropy_list)} Losses: {sum(e_sum[0])/len(e_sum[0])}, LR:{curr_lr} Patience: {patience}")
        return sum(cross_entropy_list)/len(cross_entropy_list), [sum(e_sum[0])/len(e_sum[0])]
    else:
        print(f"Epoch: {epoch} Avg_{mode}_Loss: {sum(cross_entropy_list)/len(cross_entropy_list)} Losses: {sum(e_sum[0])/len(e_sum[0]):},{sum(e_sum[1])/len(e_sum[1])}  LR:{curr_lr} Patience: {patience}")
        return sum(cross_entropy_list)/len(cross_entropy_list), [sum(e_sum[0])/len(e_sum[0]),sum(e_sum[1])/len(e_sum[1])]
    
def testing(model, epoch, data_loader, device, nclasses=1, iteration=0, path=""):
    l=[]
    p=[]
    model.eval()
    class_accuracy=[]
    with torch.no_grad():     
        for idx,(info) in enumerate(tqdm(data_loader)):
            image, label = info
            image = image.to(device)
            label = label.to(device)
            preds= model(image)
            preds= torch.softmax(preds.squeeze(), dim=0)
            preds = torch.argmax(preds, dim=0)
            image = image.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            l.extend([label.item()])
            p.append(preds.item())
            del image
            del preds 
            del label
        acc= accuracy_score(np.asarray(l), np.asarray(p))
        a= classification_report(np.asarray(l), np.asarray(p), output_dict=True)
        classes= [str(i) for i in range(nclasses)]
        for i in a.keys():
            if i in classes:
                class_accuracy.append(a[i]['f1-score'])
    print(f"Epoch: {epoch} Global_Accuracy: {acc}")           
    return acc, class_accuracy

def testing_sigmoid(model, epoch, data_loader, device, nclasses=1, iteration=0, path=""):
    # l=[[],[],[],[],[],[],[]]
    # p=[[],[],[],[],[],[],[]]
    l=[]
    p=[]
    for i in range(nclasses):
        l.append([])
        p.append([])

    model.eval()
    class_accuracy=[]
    with torch.no_grad():     
        # for idx,(img_b, label,_) in enumerate(tqdm(data_loader)):
        for idx,(info) in enumerate(tqdm(data_loader)):
            if len(info) > 2:
                img_b, label, masks, img_a= info
                masks = masks.to(device)
                img_a= img_a.to(device)
            else:
                img_b, label = info
            img_b = img_b.to(device)
            label = label.to(device)
            preds= model(img_b)
            preds= torch.sigmoid(preds)
            img_b= img_b.detach().cpu().numpy()
            label= label.detach().cpu().numpy().squeeze()
            preds= preds.detach().cpu().numpy().squeeze()
            

            for i in range(nclasses):
                l[i].append(label[i])
                p[i].append(np.round(preds[i]))

            del img_b
            del label
            del preds

        for i in range(nclasses):
          class_accuracy.append(accuracy_score(np.asarray(l[i]), np.asarray(p[i])))
            
        # tm = MetricsClass(input_shape=(7, 1024, 1))
        # dt = DataProcessing(Data="VectorLabel")
            
        # #? Y label
        # listTrueKinases = dt.splitVectorLabelV2(predictedLabel=y_test, numOutput=20)
        # #? Predicted label
        # listPredictedKinasesLoss = dt.splitVectorLabelV2(predictedLabel=pred2, numOutput=20)                
        # #! nuove metriche 
        # #/Calcolo le metriche
        # modelType = ["LossModel", "AccModel"]
        # for kinLoss in range(len(listTrueKinases)):
        #     #/ Calcolo le metriche 
        #     acc, los, sensitivity, zero_accuracy, MCC, roc_auc, f1, confusion, balanced, summaryRes = tm.metrics(yTrue=listTrueKinases[kinLoss], yPred=listPredictedKinasesLoss[kinLoss])
        #     #/Salvo i risultati
        #     tm.writeResults(summaryResults=summaryRes, confusion=confusion, resultsPath=path_results, tuningResultsPath=path_results, fileName=f"Kinase_{kinLoss+1}.csv", modelName=modelType[idx], combination=0, kinaseNumber=kinLoss+1, parameter=["See JsonFile"], fold=num_prova)
        #     #/Creo la directory per l'enrichment factor


    print(f"Epoch: {epoch} Global_Accuracy: {sum(class_accuracy)/len(class_accuracy)}")           
    return sum(class_accuracy)/len(class_accuracy), class_accuracy

def training(iterations, model, path, lr, best_loss, last_epoch, epochs, train_loader, test_loader, val_loader, device, patience_max=50, criterion= nn.BCELoss, nclasses=10, optimizer=None, type="classificazione"):
    patience=0
    os.makedirs(path, exist_ok=True)
    training_loss = []
    training_entropia = []
    validation_loss=[]
    validation_entropia=[]
    best_entropy=9999
    optimizer= optimizer
    best_accuracy = 0.
    accuracy = 0.
    class_accuracy= [1. for i in range(nclasses)]
    for e in range(last_epoch,epochs):
        flag= "None"
        print(f'Epoch:{e}/{epochs}')
        train_loss, train_entr= train_new(model, lr, e, train_loader,device, patience, criterion, optimizer=optimizer, type=type, iteration=iterations)
        val_loss, val_entr = train_new(model=model, epoch=e, train_loader=val_loader, device=device, type=type, iteration=iterations, train=False, patience=patience, criterion=criterion, lr=lr)
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        training_entropia.append(train_entr)
        plot_double_graph(training_loss, validation_loss, path, ['Train_Loss', 'Val_Loss'], f'losses_{iterations}', e, iterations)
        if val_loss < best_loss:
            flag=" Best Loss "
            best_loss=val_loss
            print("We have a new best result in Loss!!")
            patience=0
            torch.save(model,f"{path}/model&weights_{iterations}_best_loss.pth")
            torch.save(model.state_dict(),f"{path}/weights_{iterations}_best_loss.pth")
        else:
            patience +=1
            if patience == patience_max:
                return best_loss, best_entropy
        #/ checkpoint accuracy 
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            acc, f1_score_list = testing_sigmoid(model, e, test_loader, device=device, nclasses=nclasses, iteration=iterations, path=path)
        else:
            acc, f1_score_list = testing(model, e, test_loader, device=device, nclasses=nclasses, iteration=iterations, path=path)
        if acc > best_accuracy:
            flag= " Best Accuracy "
            best_accuracy = acc
            print("We have a new best Accuracy!!")
            torch.save(model,f"{path}/model&weights_{iterations}_best_accuracy.pth")
            torch.save(model.state_dict(),f"{path}/weights_{iterations}_best_accuracy.pth")
        f = open(f"{path}/log.txt", 'a')
        f.write(f'\nEpoch: {e}, Iterations:{iterations}, Loss: {val_loss}, Entropia: {val_entr}, Accuracy:{acc}, f1_score_list:{f1_score_list}, flag:{flag}')
        f.close()
    return best_loss, best_entropy

class Dataset_v2(Dataset):
    def __init__(self, root, csv, mask=False, transform=None, test=False):
        super(Dataset_v2, self).__init__()
        self.root = root
        self.transform = transform
        self.csv = csv
        self.mask= mask
        self.test= test
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
               

            
        image_index= self.csv.loc[index, 'ID']
        label = np.asarray(self.csv.iloc[index, 1:].values, dtype=np.float32)
        img = cv2.imread(f"{image_index}.jpg")
                
        if self.transform:
            img = self.transform(img)
        
        if self.test:
            if self.root is not None:
                return img, torch.tensor(label, dtype=torch.float32), os.path.join(self.root, f"{image_index}.png")
        
            else:
                return img, torch.tensor(label, dtype=torch.float32),f"{image_index}.jpg"

        
        return img, torch.tensor(label, dtype=torch.int64)