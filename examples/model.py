
import random
from tkinter.font import names
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
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from skimage import io, transform as sktransform
import os
from pretrained import alex,densenet, inception, resnet
# from label_data import IMAGE_PATH
from sklearn.model_selection import GridSearchCV
from random_model import return_valid_frame
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset


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
        # image = io.imread(img_name, as_gray=False)

        image = Image.open(img_name).convert('RGB')
        # image = sktransform.resize(image,(24,28))
        # image = image.astype('float32')

        # One-hot encoded label from 2nd col onwards
        emo_lab = self.music_frame.iloc[idx, 2:]
        emo_lab = np.array([emo_lab])
        emo_lab = emo_lab.astype('float32')
        sample = {'image': image, 'emo_lab': emo_lab}

        if self.transform:
            sample['image'] = (self.transform((sample['image'])),img_name)
            # sample['image'] = sample['image'].unsqueeze(0)

        return sample

# Model Definition

class Net(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 128, 5)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(128, 512, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 2)
        self.batchnorm5 = nn.BatchNorm2d(512)


        self.pool = nn.MaxPool2d(2,2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # x = self.batchnorm1(F.relu(self.pool(self.conv1(x))))
        # x = self.batchnorm2(F.relu(self.pool(self.conv2(x))))
        # x = self.batchnorm3(F.relu(self.pool(self.conv3(x))))
        # x = self.batchnorm4(F.relu(self.pool(self.conv4(x))))
        # x = self.batchnorm5(F.relu(self.pool(self.conv5(x))))
        # x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = self.flatten(x)
        # x = self.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(x)
        return x


def random_accuracy_check():  

    # Setting target
    labels = np.zeros(62)
    labels[:5] = 1
    np.random.shuffle(labels)

    #Setting random predictions
    rand_arr = np.zeros(62)
    rand_arr[:5] = 1

    for TRIALS in [10,100,1000,10000,100000]:    
        ac_score = 0
        for i in range(TRIALS):  
            np.random.shuffle(rand_arr)
            correct = len([index for index, (e1, e2) in enumerate(zip(labels, rand_arr)) if e1 == e2 and e1 == 1])
            # new_labels = torch.Tensor(labels).unsqueeze(0)
            # new_rand_arr = torch.Tensor(rand_arr).unsqueeze(0)
            # print((new_labels == new_rand_arr))
            # acc = (new_labels == new_rand_arr).all(dim=1).float().mean()
            # print(acc)
            acc = correct/5
            ac_score += acc

        print(f"Random Set accuracy for {TRIALS} trials = {(ac_score/TRIALS)*100}")

    # Math based accuracy
    # favourable = 5c5*1 + 5c4.59c1.0.8 + 5c3.59c2.0.6 + 5c2.59c3.0.4 + 5c1.59c4.0.2 + 59c5.0.0
    # total = 64c5
    # accurate = (favourable / total)*100

    # print(f"Random set accuracy calculated using formula = {accurate}")

def model_training(net, device, train_dataloader, val_dataloader, config, walk):
    
    # Declare/Initialize model
    # net = Net(labels=LABELS)


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
# f"/home/as14034/torchMoji/t257654.ckpt

    # net.eval()
    
    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in val_dataloader:
    #         # print("Batching")
    #         images, labels = data['image'], data['emo_lab']
    #         images, _ = images
    #         labels = labels.squeeze(dim=1)
    #         # outputs = net(images.float())
    #         # print(images.shape)
    #         outputs = net(images)
    #         # outputs = torch.nn.functional.sigmoid(outputs)


    #         # print("Predictions = ",outputs)
    #         # print("True = ", labels)


    #         _, pred_ind = torch.topk(outputs,5)
    #         _, tru_ind = torch.topk(labels,5)

    #         # print(f"Top predictions {pred_ind}, {tru_ind}")
            
    #         batch_acc = 0
    #         for i in range(tru_ind.shape[0]):
    #             overlap = np.intersect1d(tru_ind[i], pred_ind[i])
    #             batch_acc += len(overlap)/5

    #         # print(pred_ind)
    #         # print(tru_ind)
            
            
    #         score = (tru_ind == pred_ind).float().mean()

    #         total += score
    #         correct += batch_acc/tru_ind.shape[0]


    # print(f'Mean accuracy of the network on the test set (index based) is {(total / len(test_dataloader))* 100}%')
    # print(f'Mean accuracy of the network on the test set (intersection based) is {(correct / len(test_dataloader))* 100}%')



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
            # print("Batching")
            images, labels = data['image'], data['emo_lab']
            images, name = images

            # print(name)
            # print(type(images))
            labels = labels.squeeze(dim=1)
            # outputs = net(images.float())
            # print(images.shape)
            outputs = net(images)
            # outputs = torch.nn.functional.sigmoid(outputs)


            # print("Predictions = ",outputs)
            # print("True = ", labels)


            _, pred_ind = torch.topk(outputs,topk)
            _, tru_ind = torch.topk(labels,topk)

            # print(f"Top predictions {pred_ind}, {tru_ind}")
            batch_acc = 0
            # print(tru_ind.shape[0], len(name))
            
            for i in range(tru_ind.shape[0]):
                overlap = np.intersect1d(tru_ind[i], pred_ind[i])
                dataframe.loc[counter] = [name[i],tru_ind[i].tolist(),pred_ind[i].tolist()]
                counter += 1
                # print(len(overlap)/topk)
                batch_acc += len(overlap)/topk

            # print(pred_ind)
            # print(tru_ind)
            
            # print(f"Batch accuracy = {batch_acc/tru_ind.shape[0]}")
            # print()
            score = (tru_ind == pred_ind).float().mean()

            total += score
            correct += batch_acc/tru_ind.shape[0]

    # print(dataframe.shape)
    # dataframe.to_csv('testset_labels.csv', index=False)
    print(f'Mean accuracy of the network on the test set (index based) for {topk} is {(total / len(test_dataloader))* 100}%')
    print(f'Mean accuracy of the network on the test set (intersection based) for {topk} is {(correct / len(test_dataloader))* 100}%')

    # return (correct / len(test_dataloader))

if __name__ == '__main__':

    # random_accuracy_check()
    device = torch.device('mps')
    # device='cpu'

    # '/scratch/as14034/emoji/images/train'
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
    

    # for walk in range(5):
    #     config = {}
    #     config['max_epochs'] = random.choice([10,20,30,50])
    #     config['batch_size'] = random.choice([64,128,256,512])
    #     config['lr'] = random.choice([0.1,0.01,0.001])
        
    #     # Define new model for each trial
    net = resnet(model)
    config = {'batch_size': 512, 'max_epochs': 50, 'lr': 0.01}
    # # #     # Define Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    model_training(net, device, train_dataloader,val_dataloader,config,'train')

    # net = Net(labels=LABELS)


    # print(sum([param.nelement() for param in net.parameters()]))

    # model_training(net, device, train_dataloader, 0)
    # model_test(densenet(model),"/Users/amanshukla/miniforge3/torchMoji/model/train_dense5.ckpt", test_dataloader,5)
    
    # for k in range(0,26,5):
    # res = f"/Users/amanshukla/miniforge3/torchMoji/model/train.ckpt"    
    # model_test(inception(model), res, test_dataloader,5)
    # acc = []
    # for i in range(1,16):
    #     acc.append(model_test(densenet(model), res, test_dataloader,i))

    # plt.plot(acc)
    # model_test(inception(model), res, test_dataloader,15)
    # plt.legend()
    # plt.savefig('curve10.png')
    # plt.show()


    

    # model = NeuralNetClassifier(net, train_split=None)
    # cv = GridSearchCV(model, param)

    # cv.fit(train_dataset)
    # print(model.get_params().keys())
    # print(cv.get_params().keys())
    # X = SliceDataset(train_dataset,idx=0)
    # y = SliceDataset(train_dataset,idx=1)
    # print(type(y))

    # x_train = [element['image'] for element in train_dataset]
    # y_train = [element['emo_lab'] for element in train_dataset]

    # # # print(len(y_train))
    # x_train = np.stack(x_train,axis=0)
    # y_train = np.stack(y_train,axis=0)

    # print(x_train.shape)
    # print(y_train.shape)
    # for i in range(len(train_dataset)):    
    #     print(type(train_dataset[i]['emo_lab']))
    # cv.fit(x_train,y=y_train)
    # print(cv.best_score_, cv.best_params_)

    # torch.hub.set_dir('/scratch/as14034/')