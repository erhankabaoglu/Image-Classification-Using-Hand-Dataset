from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, util
import numpy as np
import matplotlib.pyplot as plt
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import glob
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import h5py


warnings.filterwarnings("ignore")
plt.ion()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(torch.cuda.get_device_name(0))
cudnn.bgenchmark = True


class HandDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.hand_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.h_dataset = None
        with h5py.File(self.root_dir, 'r') as file:
            self.len_d = len(file)

    def __len__(self):
        return self.len_d

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.hand_frame.iloc[idx, 7]
        if self.h_dataset is None:
            self.h_dataset = h5py.File(self.root_dir, 'r')
        image = self.h_dataset[img_name][()]
        gender = self.hand_frame.iloc[idx, 2]
        aspectOfHand = self.hand_frame.iloc[idx, 6]
        sample = {'image': image, 'label': gender + ' ' + aspectOfHand}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def splitTrainValidationTest(dataset, batch_size):
    validation_split, test_split = .6, .8
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    v_split = int(np.floor(validation_split * dataset_size))
    t_split = int(np.floor(test_split * dataset_size))
    np.random.seed(10)
    np.random.shuffle(indices)
    train_idx, validation_idx, test_idx = indices[:v_split], indices[v_split:t_split], indices[t_split:]

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4)

    return train_loader, validation_loader, test_loader


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class CNNNet(nn.Module):
    def __init__(self, num_classes=8, p=0.0):
        super(CNNNet, self).__init__()

        # Create 7 layers of the unit with max pooling in between
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = Unit(in_channels=3, out_channels=8)
        self.conv2 = Unit(in_channels=8, out_channels=16)
        self.conv3 = Unit(in_channels=16, out_channels=32)
        self.conv4 = Unit(in_channels=32, out_channels=64)
        self.conv5 = Unit(in_channels=64, out_channels=128)
        self.conv6 = Unit(in_channels=128, out_channels=128)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.conv1, self.conv2, self.conv3, self.pool1, self.conv4, self.conv5, self.conv6,
                                 self.pool1)

        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)
        self.drop = nn.Dropout(p=p)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.net(input)
        output = output.reshape(-1, 128 * 8 * 8)
        output = self.drop(self.relu(self.fc1(output)))
        output = self.drop(self.relu(self.fc2(output)))
        output = self.fc3(output)
        return output


class CNNNet2(nn.Module):
    def __init__(self, num_classes=8, p=0.0):
        super(CNNNet2, self).__init__()

        # Create 7 layers of the unit with max pooling in between
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = Unit(in_channels=3, out_channels=8)
        self.conv2 = Unit(in_channels=8, out_channels=8)
        self.conv3 = Unit(in_channels=8, out_channels=8)

        self.conv4 = Unit(in_channels=8, out_channels=16)
        self.conv5 = Unit(in_channels=16, out_channels=16)
        self.conv6 = Unit(in_channels=16, out_channels=16)

        self.conv7 = Unit(in_channels=16, out_channels=32)
        self.conv8 = Unit(in_channels=32, out_channels=32)
        self.conv9 = Unit(in_channels=32, out_channels=32)

        self.conv10 = Unit(in_channels=32, out_channels=64)
        self.conv11 = Unit(in_channels=64, out_channels=64)
        self.conv12 = Unit(in_channels=64, out_channels=64)

        self.conv13 = Unit(in_channels=64, out_channels=128)
        self.conv14 = Unit(in_channels=128, out_channels=128)
        self.conv15 = Unit(in_channels=128, out_channels=128)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.conv1, self.conv2, self.conv3, self.pool1, self.conv4, self.conv5, self.conv6,
                                 self.pool1, self.conv7, self.conv8, self.conv9,
                                 self.pool1, self.conv10, self.conv11, self.conv12, self.pool1, self.conv13,
                                 self.conv14, self.conv15, self.pool1)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=num_classes)
        self.drop = nn.Dropout(p=p)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.drop(self.relu(self.fc1(output)))
        output = self.drop(self.relu(self.fc2(output)))
        output = self.drop(self.relu(self.fc3(output)))
        output = self.fc4(output)
        return output


def adjust_dropout(epoch, train_acc, validation_acc): # it jsut used for first experiment
    """if epoch > 900 and validation_acc <= train_acc*(0.99):
        net.drop.p = 0.8
    elif epoch > 700 and validation_acc <= train_acc*(0.99):
        net.drop.p = 0.6
    elif epoch > 500 and validation_acc <= train_acc*(0.99):
        net.drop.p = 0.5
    elif epoch > 300 and validation_acc <= train_acc*(0.995):
        net.drop.p = 0.3"""
    if epoch > 200 and validation_acc <= train_acc * (0.995):
        net.drop.p = 0.2
    elif epoch > 100 and validation_acc <= train_acc * (0.995):
        net.drop.p = 0.1


def testNN(cnnNetwork, lossFunction, loader):
    with torch.no_grad():
        cnnNetwork.eval()
        total = 0.0
        correct = 0.0
        running_loss = 0.0
        for data in loader:
            images, labels = data['image'].to(device, dtype=torch.float), data['label']
            target = torch.tensor([classes.index(labels[i]) for i in range(len(labels))], dtype=torch.long).to(device)

            outputs = cnnNetwork(images)
            loss = lossFunction(outputs, target)
            running_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (prediction == target).sum().item()

        average_loss = running_loss / len(loader)

        return 100 * correct / total, average_loss


def train(cnnNetwork, lossFunction, optimizer, epoch, PATH, lrScheduler=None):
    best_acc = 0.0
    train_acc_L = []
    val_acc_L = []
    train_l_L = []
    validation_l_L = []
    total_train_loss_val = []
    for k in range(epoch):
        cnnNetwork.train()
        r_loss = 0.0
        train_ac = 0.0
        total = 0
        loss_values = []
        for i, sample_batched in enumerate(train_loader):
            input_images, labels = sample_batched['image'].to(device, dtype=torch.float), sample_batched['label']
            target = torch.tensor([classes.index(labels[i]) for i in range(len(labels))], dtype=torch.long).to(device)

            optimizer.zero_grad()

            outputs = cnnNetwork(input_images)
            loss = lossFunction(outputs, target)
            loss.backward()
            optimizer.step()

            r_loss += loss.item()
            loss_values.append(loss.item())
            _, prediction = torch.max(outputs.data, 1)
            train_ac += torch.sum(prediction == target)
            total += target.size(0)

        if lrScheduler:
            lrScheduler.step()
        tr_acc = 100 * train_ac / total
        validation_acc, validation_loss = testNN(cnnNetwork, lossFunction, validation_loader)

        # adjust_dropout(k,tr_acc,validation_acc)
        # adjust_learning_rate(k)

        train_acc_L.append(tr_acc)
        val_acc_L.append(validation_acc)
        train_l_L.append(r_loss / len(train_loader))
        validation_l_L.append(validation_loss)
        total_train_loss_val[len(total_train_loss_val):] = loss_values

        """plt.plot(np.array(loss_values), label='Train Loss')
        plt.title('Epoch: ' + str(k + 1))
        plt.xlabel('Loss number')
        plt.ylabel('Loss Values')
        plt.legend()
        plt.savefig('drive/My Drive/partical_loss ' + str(k + 1) + '.png')
        plt.show()"""

        np.save('drive/My Drive/train_acc_Lres4.npy', np.array(train_acc_L))
        np.save('drive/My Drive/val_acc_Lres4.npy', np.array(val_acc_L))
        np.save('drive/My Drive/train_l_Lres4.npy', np.array(train_l_L))
        np.save('drive/My Drive/validation_l_Lres4.npy', np.array(validation_l_L))
        np.save('drive/My Drive/total_train_loss_valres4.npy', np.array(total_train_loss_val))

        if validation_acc > best_acc:
            torch.save(cnnNetwork.state_dict(), PATH)
            best_acc = validation_acc

        print('Epoch: %d Train accuracy: %.3f Validation accuracy: %.3f' % (k + 1, tr_acc, validation_acc))


def test(cnnNetwork, loader):
    with torch.no_grad():
        cnnNetwork.eval()
        total = 0.0
        correct = 0.0
        conf_true = []
        conf_pred = []
        for data in loader:
            images, labels = data['image'].to(device, dtype=torch.float), data['label']
            target = torch.tensor([classes.index(labels[i]) for i in range(len(labels))], dtype=torch.long).to(device)

            outputs = cnnNetwork(images)
            _, prediction = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (prediction == target).sum().item()
            conf_true[len(conf_true):] = target.cpu().numpy()
            conf_pred[len(conf_pred):] = prediction.cpu().numpy()

        np.save('drive/My Drive/conf_true5.npy', np.array(conf_true))
        np.save('drive/My Drive/conf_pred5.npy', np.array(conf_pred))

        return  100 * correct / total


def createH5():
    images = glob.glob("drive/My Drive/Hands/*.jpg")
    with h5py.File('Hand.h5', 'w') as hf:
        for i in range(1):
            image_name = os.path.basename(images[i])
            print(image_name)
            image = io.imread(images[i])
            image = transform.resize(image,(265,256,3),anti_aliasing=True)
            image = util.img_as_ubyte(image)
            handSet = hf.create_dataset(
                name=image_name,
                data=image,
                compression="gzip",
                shape=(256, 256, 3),
                maxshape=(256, 256, 3),
                compression_opts=9
            )


def plotTotalLossResults(data_name1,title,save_path):
    data1 = np.load('drive/My Drive/'+data_name1)
    plt.plot(data1, label='Total Train Loss')
    plt.xlabel('Loss Number')
    plt.ylabel('Loss values')
    plt.title(title)
    plt.legend()
    plt.savefig('drive/My Drive/'+save_path)
    plt.show()


def plotLossResults(data_name1,data_name2,save_path):
    data1 = np.load('drive/My Drive/'+data_name1)
    data2 = np.load('drive/My Drive/'+data_name2)
    plt.plot(data1, label='Train Loss')
    plt.plot(data2, label='Validation Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss values')
    plt.legend()
    plt.savefig('drive/My Drive/'+save_path)
    plt.show()


def plotAccuracyResults(data_name1,data_name2,save_path):
    data1 = np.load('drive/My Drive/'+data_name1, allow_pickle=True)
    data2 = np.load('drive/My Drive/'+data_name2)
    plt.plot(data1, label='Train Accuracy')
    plt.plot(data2, label='Validation Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('drive/My Drive/'+save_path)
    plt.show()

#*************************************************************************

classes = ('male dorsal left', 'male dorsal right', 'male palmar left', 'male palmar right',
           'female dorsal left', 'female dorsal right', 'female palmar left', 'female palmar right')

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

hand_dataset = HandDataset('drive/My Drive/HandInfo.csv', 'drive/My Drive/Hand.h5', transform=transform)
batch_size = 16

train_loader, validation_loader, test_loader = splitTrainValidationTest(hand_dataset, batch_size)

net = CNNNet()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

PATH = 'drive/My Drive/hand_net7.pth'

train(net,criterion,optimizer,200,PATH)

print("Finish training")


#*************************************************************************


plotTotalLossResults('total_train_loss_valres4.npy','100 Epoch','total_train_loss_valres4.jpg')
plotAccuracyResults('train_acc_Lres4.npy','val_acc_Lres4.npy','train_validation_acc_Lres4.jpg')
plotLossResults('train_l_Lres4.npy','validation_l_Lres4.npy','train_validation_l_Lres4.jpg')


y_true = np.load('drive/My Drive/conf_true5.npy', allow_pickle=True)
y_pred = np.load('drive/My Drive/conf_pred5.npy', allow_pickle=True)
confusion_matrix(y_true, y_pred)

#*************************************************************************

resnet_18 = models.resnet18(pretrained=True)

"""for param in resnet_18.parameters():# res1
    param.requires_grad = False"""

"""for i,child in enumerate(resnet_18.children()):
    if i < 7:
        for param in child.children():
            param.requires_grad = False"""  # res2

num_ftrs = resnet_18.fc.in_features
# resnet_18.fc = nn.Linear(num_ftrs, 8) res1-2-4

"""resnet_18.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs), 
                             nn.ReLU(),
                             nn.Linear(num_ftrs, 64),
                             nn.ReLU(),
                             nn.Linear(64, 8),
                             )"""  # res3

for i, child in enumerate(resnet_18.children()):
    if i < 6:
        for param in child.children():
            param.requires_grad = False  # res4

resnet_18.fc = nn.Linear(num_ftrs, 8)

resnet_18.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(filter(lambda p: p.requires_grad, resnet_18.parameters()), lr=0.001, momentum=0.9,
                      weight_decay=0.001)

lrScheduler = lr_scheduler.StepLR(optimizer, 30)

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

hand_dataset = HandDataset('drive/My Drive/HandInfo.csv', 'drive/My Drive/Hand.h5', transform=transform)
batch_size = 64
train_loader, validation_loader, test_loader = splitTrainValidationTest(hand_dataset, batch_size)

PATH = 'drive/My Drive/hand_netres4.pth'

train(resnet_18, criterion, optimizer, 100, PATH, lrScheduler)
