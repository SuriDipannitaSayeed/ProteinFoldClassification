import argparse
import glob
import os
import time
import pickle
import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=256, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
#parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=500, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.set_device(args.device)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

with open("/content/filelist.txt","rb") as f:
    dataset = pickle.load(f,encoding="latin1")
print(dataset)
#dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)

args.num_classes = 15
args.num_features = 2

print(args)

num_training = int(len(dataset) * 1.0)
num_val = int(len(dataset) * 0.2)
print(num_val)
num_test = len(dataset) - (num_training + num_val)

#training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
training_set=dataset
validation_set=dataset[:20]
test_set=dataset[25:35]
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)

print(len(train_loader.dataset))
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True)
print(len(val_loader.dataset))
test_loader = DataLoader(test_set, batch_size=2, shuffle=True)
print(len(test_loader.dataset))
model = Model(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    min_loss = 0
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0
    acc_trainset=[]
    acc_testset=[]
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            #data=data[0]
                        
            data = data.to(args.device)
            #i = i.view(args.batch_size)
            out = model(data)
            #print(out)
            #print(data.y)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            #optimizer.zero_grad()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq((data.y)).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        test_acc, test_loss = compute_test(test_loader)
        acc_trainset.append(acc_train)
        acc_testset.append(test_acc)
    #print('acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t),)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train),
              'Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))
    test_acc, test_loss = compute_test(test_loader)
    plt.title('train set accuracy')
    plt.plot(acc_trainset)
    plt.show()
    plt.title('Test set accuracy')
    plt.plot(acc_testset)
    plt.show()
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    # Model training
    best_model = train()
    # Restore best model for test set
    #model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    test_acc, test_loss = compute_test(test_loader)
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))

