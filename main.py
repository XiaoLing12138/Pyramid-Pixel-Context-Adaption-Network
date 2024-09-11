import os
import csv
import torch
import random
import shutil

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from data.dataloaders.dataloaders_isic import ISICDataloader

import yaml
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from torch.autograd import Variable

from utils import *

with open('config.yaml', 'rb') as f:
    config = yaml.safe_load(f.read())

image_size = config["train"]["size"]
resume = config["train"]["resume"]
random_seed = config["train"]["seed"]
basic_learning_rate = config["train"]["lr"]
batch_size = config["train"]["batch-size"]
image_number = config["train"]["img_num"]
epochs = config["train"]["epochs"]
decay = config["train"]["decay"]
dataset = config["train"]["dataset"]
num_worker = config["train"]["num_worker"]
data_path = config["train"]["data_path"]
num_class = config["train"]["num_classes"]

caller = config["model"]["caller"]
model_name = config["model"]["name"]
file_name = config["model"]["file"]
checkpoint_folder = config["model"]["folder"]

log_folder = config["log"]["folder"]

exec('from models.{} import *'.format(file_name))

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

best_acc = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(config['train']['cuda'])
use_cuda = torch.cuda.is_available()

if not os.path.isdir(checkpoint_folder):
    os.mkdir(checkpoint_folder)

if not os.path.isdir(log_folder):
    os.mkdir(log_folder)

shutil.copyfile(os.path.join(sys.path[0], 'config.yaml'), os.path.join(sys.path[0], log_folder,
                                                                       model_name + '_' + str(random_seed) +
                                                                       '_config.log'))

log_file = os.path.join(sys.path[0], log_folder, model_name + '_' + str(random_seed) + '.csv')
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        log_writer = csv.writer(f, delimiter=',')
        log_writer.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])


def save_checkpoint(model, acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': model,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    torch.save(state, os.path.join(sys.path[0], checkpoint_folder,
                                   model_name + '_' + str(random_seed) + '.pth'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = basic_learning_rate
    if epoch <= 9:
        # warm-up training for large minibatch
        lr = basic_learning_rate + basic_learning_rate * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, epoch, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_index = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_index = batch_idx

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / batch_index, 100. * correct / total


def valid(model, valid_loader, valid_loader_train, epoch, criterion):
    global best_acc
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        for batch_idx, (inputs, targets) in enumerate(valid_loader_train):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(valid_loader_train), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(model, acc, epoch)
    return valid_loss / batch_idx, 100. * correct / total


def classify_report(result):
    result = result.split()
    x = result.index('accuracy')
    n = (x + 1) // 5 - 1
    tag = result[:4]
    m = {}
    for i in range(n):
        start = result.index(str(i)) + 1
        for j in range(4):
            m["grade" + str(i) + "_" + tag[j]] = result[start + j]
    m[result[x]] = result[x + 1]
    m_x = result.index("macro")
    for i in range(3):
        m[result[m_x] + "_" + tag[i]] = result[m_x + 2 + i]
    w_x = result.index("weighted")
    for i in range(3):
        m[result[w_x] + "_" + tag[i]] = result[w_x + 2 + i]
    m["total"] = result[-1]
    return m


# Testing
def test(test_loader, criterion):
    print("begin")
    global best_acc
    model_path = os.path.join(sys.path[0], checkpoint_folder, model_name + '_' + str(random_seed) + '.pth')
    checkpoint = torch.load(model_path)
    model = checkpoint['net']
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    torch.set_rng_state(checkpoint['rng_state'])

    if use_cuda:
        model.cuda()
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.deterministic = True
        print('Using CUDA..')
    model.eval()
    print(epoch)
    valid_loss = 0
    correct = 0
    total = 0
    labels = []
    pre_labels = []
    auc_output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            auc_output.extend(outputs.data.cpu().numpy())
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            labels.extend(targets.data.cpu())
            pre_labels.extend(predicted.data.cpu())
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    labels = np.reshape(labels, [-1, 1])
    pre_labels = np.reshape(pre_labels, [-1, 1])
    print(classification_report(labels, pre_labels, digits=5))

    # label -->
    auc_labels = np.zeros(shape=(len(labels), config["train"]["num_classes"]))
    for i in range(len(labels)):
        auc_labels[i][labels[i]] = 1

    auc_output = np.array(auc_output)

    auc_macro_ovo = roc_auc_score(auc_labels, auc_output, average='macro', multi_class='ovo')
    auc_macro_ovr = roc_auc_score(auc_labels, auc_output, average='macro', multi_class='ovr')
    auc_weighted_ovo = roc_auc_score(auc_labels, auc_output, average='weighted', multi_class='ovo')
    auc_weighted_ovr = roc_auc_score(auc_labels, auc_output, average='weighted', multi_class='ovr')

    with open(log_file, 'a') as f:
        f.write(classification_report(labels, pre_labels, digits=5))
        f.write("epoch: ")
        f.write(str(epoch))
        f.write("\n")
        f.write("valid_acc: ")
        f.write(str(best_acc))
        f.write("\n")
        f.write("kappa: ")
        f.write(str(cohen_kappa_score(labels, pre_labels)))
        f.write("\n")
        f.write("auc_macro_ovo: ")
        f.write(str(auc_macro_ovo))
        f.write("\n")
        f.write("auc_macro_ovr: ")
        f.write(str(auc_macro_ovr))
        f.write("\n")
        f.write("auc_weighted_ovo: ")
        f.write(str(auc_weighted_ovo))
        f.write("\n")
        f.write("auc_weighted_ovr: ")
        f.write(str(auc_weighted_ovr))


def main():
    global best_acc, dataset

    dataloader = ISICDataloader(
        batch_size=batch_size,
        num_workers=num_worker,
        img_resize=image_size,
        root_dir=data_path,
    )

    (test_loader, test_dataset,) = dataloader.run("test")

    (label_loader, label_dataset,) = dataloader.run("labeled")

    # model initial
    model = eval('{}(num_classes=num_class)'.format(caller))
    model = torch.nn.DataParallel(model).cuda()
    cudnn.deterministic = True
    print('Using', torch.cuda.device_count(), 'GPUs.')
    print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=basic_learning_rate, momentum=0.9, weight_decay=decay)

    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(os.path.join(sys.path[0], checkpoint_folder)), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(sys.path[0], checkpoint_folder,
                                             model_name + '_' + str(random_seed) + '.pth'))
        model = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model...')
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc = train(model, label_loader, epoch, criterion, optimizer)
        valid_loss, valid_acc = valid(model, test_loader, test_loader, epoch, criterion)
        with open(log_file, 'a') as f:
            log_writer = csv.writer(f, delimiter=',')
            log_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    test(test_loader, criterion)


if __name__ == '__main__':
    main()
