import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import time
import os
from pretrained import alex,densenet, inception, resnet
# from label_data import IMAGE_PATH
from random_model import return_valid_frame

class SpectrogramDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with emoji labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.music_frame = return_valid_frame(pd.read_csv(csv_file),root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.music_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.music_frame.iloc[idx, 0]+'.png')

        image = Image.open(img_name).convert('RGB')

        # One-hot encoded label from 2nd col onwards
        emo_lab = self.music_frame.iloc[idx, 2:]
        emo_lab = np.array([emo_lab])
        emo_lab = emo_lab.astype('float32')
        sample = {'image': image, 'emo_lab': emo_lab}

        if self.transform:
            sample['image'] = (self.transform((sample['image'])),img_name)

        return sample

def model_training(net, device, train_dataloader, val_dataloader, config, walk):

    net.to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    traintime = time.time()


    for ep in range(config['max_epochs']):  # loop over the dataset multiple times
        timeing = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            staring = time.time()

            # get the inputs; data is a list of [inputs, labels]
            inputs, label = data['image'], data['emo_lab']
            inputs, _ = inputs

            # print(label.shape)
            inputs = inputs.to(device)
            label = label.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            # outputs = torch.nn.functional.sigmoid(outputs)
            label = label.squeeze(dim=1)

            loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # if i % 100 == 0:    
        print(f'[{ep + 1}, {i + 1:5d}] loss: {running_loss/len(train_dataloader):.3f}')

        if ep % 5 == 0:        
        #     # Save model here for re-use and testing
            torch.save(net.state_dict(), f"/Users/amanshukla/miniforge3/torchMoji/model/trained_resnet18_{ep}.ckpt")

    print(f'Finished Training for {config}')
    print(f"Time for training = {time.time()-traintime}")

    # Save model here for re-use and testing
    torch.save(net.state_dict(), f"/Users/amanshukla/miniforge3/torchMoji/model/final_resnet18_{config['max_epochs']}.ckpt")


def model_test(net,path, test_dataloader, topk):
    
    net.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    # /home/as14034/torchMoji/t25.ckpt
    net.eval()
    
    dataframe = pd.DataFrame(columns=['test_image_id', 'real_labels', 'predicted_labels'])
    correct = 0
    total = 0
    counter = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            
            images, labels = data['image'], data['emo_lab']
            images, name = images

            labels = labels.squeeze(dim=1)
            
            outputs = net(images)

            _, pred_ind = torch.topk(outputs,topk)
            _, tru_ind = torch.topk(labels,topk)

            batch_acc = 0
            
            for i in range(tru_ind.shape[0]):
                overlap = np.intersect1d(tru_ind[i], pred_ind[i])
                dataframe.loc[counter] = [name[i],tru_ind[i].tolist(),pred_ind[i].tolist()]
                counter += 1
                
                batch_acc += len(overlap)/topk


            score = (tru_ind == pred_ind).float().mean()

            total += score
            correct += batch_acc/tru_ind.shape[0]

    print(f'Mean accuracy of the network on the test set (index based) for {topk} is {(total / len(test_dataloader))* 100}%')
    print(f'Mean accuracy of the network on the test set (intersection based) for {topk} is {(correct / len(test_dataloader))* 100}%')


if __name__ == '__main__':

    device = torch.device('mps')


    # Load images and emoji labels
    train_dataset = SpectrogramDataset('test_25_v4.csv','/Volumes/TOSHIBA/DALI/images/train' ,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])]))

    val_dataset = SpectrogramDataset('test_25_v4.csv','/Volumes/TOSHIBA/DALI/images/val' ,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])]))

    test_dataset = SpectrogramDataset('test_25_v4.csv','/Volumes/TOSHIBA/DALI/images/test' ,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])]))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    # Fixed emoji labels to predict
    LABELS = 64
        # Use Pretrained 
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    
    net = resnet(model)
    config = {'batch_size': 512, 'max_epochs': 50, 'lr': 0.01}
    
    # Define Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    model_training(net, device, train_dataloader,val_dataloader,config,'train')