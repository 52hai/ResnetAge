import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.stats as stats
import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import os


def file_count(local_path):
    all_file_num = 0
    file_list = os.listdir(local_path)
    for file_name in file_list:
        if not os.path.isdir(os.path.join(local_path, file_name)):
            all_file_num += 1
    return all_file_num

geo_check = [
    'GSE58045','GSE20236', 'GSE72775', 'GSE73103',
    'GSE72773', 'GSE61496', 'GSE50660', 'GSE60132',
    'GSE40279', 'GSE55763', 'GSE78874', 'GSE92767', 'GSE99029', 'GSE138279', 'GSE34035', 'GSE28746',
    'GSE101961', 'GSE15745', 'GSE56105', 'GSE94876', 'GSE137903', 'GSE137894', 'GSE94734', 'GSE137502',
    'GSE137898', 'GSE137688', 'GSE25892', 'GSE48988', 'GSE36194', 'GSE51954', 'GSE56581', 'GSE64511'
]

data_name_pre = "/home/haiboquan/data/"
ph_name_pre = "/home/haiboquan/data/"

data_labels = []
data_labels = pd.DataFrame(data_labels)
data_features = []
data_features = pd.DataFrame(data_features)

for GEO in geo_check:
    ph_name = ph_name_pre + str(GEO) + '_pheno.csv'
    data_ph = pd.read_csv(ph_name)
    Age_unit = data_ph['Age_unit'][1]
    if Age_unit == "Year":
        data_labels1 = data_ph['Age']
    elif Age_unit == "Month":
        data_labels1 = data_ph['Age'] / 12
    elif Age_unit == "Week":
        data_labels1 = data_ph['Age'] / 48
    elif Age_unit == "Day":
        data_labels1 = data_ph['Age'] / 365

    if data_labels.shape[0]==0:
        data_labels = data_labels1
    else:
        data_labels = pd.concat([data_labels,data_labels1],axis=0)

for GEO in geo_check:
    data_name = data_name_pre + str(GEO) + '_beta.csv'

    if os.path.exists(data_name):
        data_2 = pd.read_csv(data_name)
        data_2.set_index(["Unnamed: 0"], inplace=True)
        data_2 = data_2.T
        print("1",GEO,data_2.shape)
        print("2",data_features.shape)
        if data_features.shape[1]==0:
            data_features = data_2
        else:
            data_features = pd.concat([data_features,data_2],axis=0,join='inner')
        print("拼接后",data_features.shape)

    if not os.path.exists(data_name):
        local_path = data_name_pre + GEO + "_beta"
        f_count = file_count(local_path)
        print("文件夹文件个数:", f_count)
        # dat0Name = str(GEO) + '_beta_1.csv'
        # data_2 = pd.read_csv(local_path+"/"+ dat0Name)
        for i in range(1, f_count + 1):
            dat0Name = str(GEO) + '_beta_' + str(i) + '.csv'
            data_2 = pd.read_csv(data_name_pre + GEO + "_beta/" + dat0Name)
            # data_d1 = data_d1.drop(columns=["Unnamed: 0"],axis=1)
            # data_2 = pd.concat([data_2,data_d1],axis=1)
            data_2.set_index(["Unnamed: 0"], inplace=True)
            data_2 = data_2.T  # 转置
            print("1", GEO, data_2.shape)
            print("2", data_features.shape)

            if data_features.shape[1] == 0:
                data_features = data_2
            else:
                data_features = pd.concat([data_features, data_2], axis=0, join='inner')
            print("拼接后", data_features.shape)
        print("样本数", data_labels.shape)
        print("表达矩阵", data_features.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data_features.values,
    data_labels.values,
    test_size=0.375,
    random_state=1)

X_train_std = X_train
X_test_std = X_test

random_seed = 1
learning_rate = 0.1
num_epochs = 250
batch_size = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):

    def __init__(self, feature_array, label_array, dtype=np.float32):

        self.features = feature_array.astype(np.float32)
        self.labels = label_array

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        return inputs, label

    def __len__(self):
        return self.labels.shape[0]

train_dataset = MyDataset(X_train_std, y_train)
test_dataset = MyDataset(X_test_std, y_test)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

for inputs, labels in train_loader:
     print('Input batch dimensions:', inputs.shape)
     In_features = inputs.shape[1]
     print('Input label dimensions:', labels.shape)
     break
print(In_features)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Define residual blocks
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):

        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.elu(out)
        return out

class ResNet_2D(nn.Module):
    def __init__(self, block, num_blocks, input_channel):
        super(ResNet_2D, self).__init__()
        self.in_planes = 1600
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 1600, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1600))

        self.layer1 = self._make_layer(block, 1600, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 1600, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 1600, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 1600, num_blocks[3], stride=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(1600, 800, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(800)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(800, 400, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(400)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(400, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        purpose: Duplicate layer
        :param block: BasicBlock
        :param planes:
        :param num_blocks:
        :param stride:
        :return:
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.elu(self.conv1(x))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2) + out2
        out4 = self.layer4(out3) + out2 + out3
        drop = nn.Dropout2d(p=0.5, inplace=False)
        out = drop(F.elu(self.conv2(out4)))
        out = F.elu(self.conv3(out))
        out = F.relu(self.conv4(out))
        return out
def ResNet18(n):
    return ResNet_2D(BasicBlock,[2,1,1,1],n)

torch.manual_seed(random_seed)
model = ResNet18(In_features)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def recon_criterion(input, target):
    return torch.mean(torch.abs(input - target))

for epoch in range(num_epochs):
    model = model.train()
    loss_list = []
    for batch_idx, (features, class_labels) in enumerate(train_loader):
        if batch_idx == len(train_loader) - 1:
            break
        levels =  class_labels.unsqueeze(1)
        features = features.to(DEVICE)
        levels = levels.to(torch.float)
        levels = levels.to(DEVICE)
        features = features.reshape((features.shape[0], In_features, 1, 1))
        logits = model(features)
        logits = logits.squeeze(1).squeeze(1)
        loss_fn1 = torch.nn.MSELoss(reduction='mean')
        loss = loss_fn1(logits, levels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not batch_idx % 200:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                   %(epoch+1, num_epochs, batch_idx,
                     len(train_loader), loss))
            loss_list.append(loss.item())

def r2(x,y):
    return stats.pearsonr(x,y)[0] **2

def plot_known_predicted_ages(known_ages, predicted_ages, label=None):
    def func(x, a, b, c):
        return a * np.asarray(x)**0.5 + c
    popt, pcov = optimize.curve_fit(func, [1 + x for x in known_ages], predicted_ages)
    rsquared = r2(predicted_ages, func([1 + x for x in known_ages], *popt))
    plot_label = f'$f(x)={popt[0]:.2f}x^{{1/2}} {popt[2]:.2f}, R^{{2}}={rsquared:.2f}$'
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(sorted(known_ages), func(sorted([1 + x for x in known_ages]), *popt), 'r--', label=plot_label)
    ax.scatter(known_ages, predicted_ages, marker='o', alpha=0.8, color='c')
    ax.set_title(label, fontsize=18)
    ax.set_xlabel('Chronological Age', fontsize=16)
    ax.set_ylabel('Predicted Age', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16)
    plt.xlim(0, 125)
    plt.ylim(0, 125)
    plt.show()

def proba_to_label(probas):
    pass

def compute_mae_and_mse(model, data_loader, device,b):
    model = model.eval()
    with torch.no_grad():
        mae, mse, acc, num_examples = 0., 0., 0., 0
        pre_Age = []
        true_Age = []
        for i, (features, targets) in enumerate(data_loader):
            if i == len(data_loader) - 1:
                break
            features = features.to(device)
            targets = targets.float().to(device)
            features = features.reshape((features.shape[0], In_features, 1, 1))
            logits = model(features)
            logits = logits.squeeze(1).squeeze(1)
            predicted_labels = logits.float()
            predicted_labels = predicted_labels.squeeze(1)
            pre_Age += predicted_labels.tolist()
            true_Age += targets.tolist()
            num_examples += targets.size(0)
        pre_Age = np.array(pre_Age)
        true_Age = np.array(true_Age)

        if b == 1:
            plot_known_predicted_ages(true_Age, pre_Age, 'SAM-Resnet Train Predicted Ages')
        elif b==2:
            plot_known_predicted_ages(true_Age, pre_Age, 'SAM-Resnet Val Predicted Ages')
        else:
            plot_known_predicted_ages(true_Age, pre_Age, 'SAM-Resnet Test Predicted Ages')

        mae += np.sum(np.absolute(pre_Age - true_Age))
        mse += np.sum((pre_Age - true_Age)**2)
        print(num_examples)
        print(pre_Age)
        print(true_Age)
        mae = mae / num_examples
        mse = mse / num_examples
        mad = np.median(abs(pre_Age-true_Age))
        return mae, mse, mad

train_mae, train_mse, train_mad = compute_mae_and_mse(model, train_loader, DEVICE, 1)
test_mae, test_mse, test_mad = compute_mae_and_mse(model, test_loader, DEVICE, 3)

print(f'Mean absolute error (train/test): {train_mae:.2f} | {test_mae:.2f}')
print(f'Mean squared error (train/test): {train_mse:.2f} | {test_mse:.2f}')
print(f'Mean squared error (train/test): {train_mad:.2f} | {test_mad:.2f}')

