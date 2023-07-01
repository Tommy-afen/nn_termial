import torch
import torchvision
from torchvision import transforms, models
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torchtoolbox.transform import Cutout
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from copy import deepcopy
import tqdm
import argparse
import os
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='Select a gpu.')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--aug', type=str, default=None, help='Type of data augmentation.')
parser.add_argument('--epochs', type=int, default=60, help='Set the epochs for training')
parser.add_argument('--lr', type=float, default=0.01, help='Set the learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Set the weight decay rate.')
parser.add_argument('--device', type=str, default='gpu', help='Select a device.')

args = parser.parse_args()

device = None
if args.device == 'cpu':
    device = args.device
else:
    device = f'cuda:{args.gpu_id}'
    

image_transform_origin = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

image_transform_cutout = transforms.Compose([
    Cutout(scale=(1/16, 1/16), p=1.0, ratio=(1,1), value=(0,1), pixel_level=True),
    # Cutout(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
    

def train(args):
    writer = SummaryWriter()
    
    train_data = torchvision.datasets.CIFAR100(root='data', transform=image_transform_origin, train=True, download=True)
    test_data = torchvision.datasets.CIFAR100(root='data', transform=image_transform_origin, train=False, download=True)
    train_data, val_data = Subset(train_data, range(37500)), Subset(train_data, range(37500, 50000))
    
    train_size = len(train_data)
    val_size = len(val_data)
    test_size = len(test_data)
    print(train_size)
    print(val_size)
    train_loader = DataLoader(dataset=train_data, batch_size=128, num_workers=10)
    val_loader = DataLoader(dataset=val_data, batch_size=128, num_workers=10)
    test_loader = DataLoader(dataset=test_data, batch_size=128, num_workers=10)
    if args.model == 'vit':
        model = models.vision_transformer._vision_transformer(
            patch_size=16,
            num_layers=12,
            num_heads=12 // 3,
            hidden_dim=768 // 3,
            mlp_dim=3072 // 3,
            weights=models.vision_transformer.ViT_B_16_Weights.verify(None),
            progress=True,
            num_classes=100
        )
        encoderblock = list(model.encoder.layers)
        for i in range(2):
            encoderblock.append(models.vision_transformer.EncoderBlock(
                num_heads=12 // 3,
                hidden_dim=768 // 3,
                mlp_dim=3072 // 3,
                dropout=0,
                attention_dropout=0,
                norm_layer=models.vision_transformer.partial(torch.nn.LayerNorm, eps=1e-6)
            ))
        model.encoder.layers = torch.nn.Sequential(*encoderblock)
    else:
        model = models.resnet18(pretrained=False, num_classes=100)
    # fc_inputs = model.fc.in_features
    # model.fc = nn.Sequential(nn.Linear(fc_inputs, 100), nn.LogSoftmax(dim=1))
    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    loss_function_1 = nn.CrossEntropyLoss().to(device)
    loss_function_2 = SoftTargetCrossEntropy()
    
    best_val_acc = .0
    best_val_loss = .0
    best_epoch = .0
    best_model = None
    weights = None
    for epo in range(args.epochs):
        print(f'epoch:{epo}/{args.epochs}')
        
        train_loss = .0
        val_loss = .0
        train_acc = .0
        val_acc = .0
        test_acc = .0
        test_loss = .0
        
        if not args.aug:
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                model.train()
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = loss_function_1(outputs, targets)
                loss.backward()
                optimizer.step()
                
        
        if args.aug == 'cutout':
            train_data_cutout = torchvision.datasets.CIFAR100(root='data', transform=image_transform_cutout, train=True, download=True)
    
            train_data_cutout, _ = random_split(dataset=train_data, lengths=[0.8, 0.2])
            train_loader_cutout = DataLoader(dataset=train_data_cutout, batch_size=128, num_workers=10)
            for idx, (inputs, targets) in enumerate(train_loader_cutout):
                inputs, targets = inputs.to(device), targets.to(device)
                model.train()
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = loss_function_1(outputs, targets)
                loss.backward()
                optimizer.step()
            
        
        if args.aug == 'mixup':
            mixup_args = {
                'mixup_alpha': 0.5,
                'cutmix_alpha': 0.,
                'cutmix_minmax': None,
                'prob': 0.3,
                'switch_prob': 0,
                'mode': 'batch',
                'label_smoothing': 0,
                'num_classes': 100}
            mixup_fn = Mixup(**mixup_args)
            
            for idx, (inputs, targets) in enumerate(train_loader):
                targets_ori = targets.clone().to(device)
                inputs, targets = mixup_fn(inputs, targets)
                inputs, targets = inputs.to(device), targets.to(device)
                
                model.train()
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = loss_function_2(outputs, targets)
                loss.backward()
                optimizer.step()
                
                
        
                
        if args.aug == 'cutmix':
            mixup_args = {
                'mixup_alpha': 0.,
                'cutmix_alpha': 1,
                'cutmix_minmax': None,
                'prob': 1.0,
                'switch_prob': 0.,
                'mode': 'batch',
                'label_smoothing': 0,
                'num_classes': 100}
            cutmix_fn = Mixup(**mixup_args)
            
            for idx, (inputs, targets) in enumerate(train_loader):
                targets_ori = targets.clone().to(device)
                inputs, targets = cutmix_fn(inputs, targets)
                inputs, targets = inputs.to(device), targets.to(device)
                
                model.train()
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = loss_function_2(outputs, targets)
                loss.backward()
                optimizer.step()
                
                
                
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            model.eval()
            
            outputs = model(inputs)
            loss = loss_function_1(outputs, targets)
            
            _, predicts = torch.max(outputs.data, 1)
            correct_counts = targets.eq(predicts.data.view_as(targets))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            train_loss += loss.item() * inputs.size(0)
            train_acc += acc.item() * inputs.size(0)
                
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            model.eval()
            
            outputs = model(inputs)
            loss = loss_function_1(outputs, targets)
            _, predicts = torch.max(outputs.data, 1)
            correct_counts = targets.eq(predicts.data.view_as(targets))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            val_loss += loss.item() * inputs.size(0)
            val_acc += acc.item() * inputs.size(0)
            
        
        
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            model.eval()
            
            outputs = model(inputs)
            loss = loss_function_1(outputs, targets)
            _, predicts = torch.max(outputs.data, 1)
            correct_counts = targets.eq(predicts.data.view_as(targets))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            test_loss += loss.item() * inputs.size(0)
            test_acc += acc.item() * inputs.size(0)
            

        
            
        val_acc /= float(val_size)
        val_loss /= float(val_size)
        
        train_acc /= float(train_size)
        train_loss /= float(train_size)
        
        test_acc /= float(test_size)
        test_loss /= float(test_size)
        writer.add_scalars(f"{args.aug if args.aug else 'non-aug'}/{args.model}/loss", {'train_loss': train_loss, 'val_loss': val_loss},
                          epo + 1)
        writer.add_scalars(f"{args.aug if args.aug else 'non-aug'}/{args.model}/acc", {'train_acc': train_acc, 'val_acc': val_acc},
                          epo + 1)
        

        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epo + 1
            weights = model.state_dict()
            
        print('training loss:%-7.2f training accuracy:%-7.2f \nvalid loss:%-7.2f valid accuracy:%-7.2f\ntest loss:%-7.2f test accuracy:%-7.2f' % (train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
        print('BEST EPOCH:%.d\nBest valid loss:%-7.2f Best valid accuracy:%-7.2f' % (best_epoch, best_val_loss, best_val_acc))
        
    model.load_state_dict(weights)
    
    if not args.aug:
        torch.save(model, f'models_save/{args.model}.pt')
    else:
        torch.save(model, f'models_save/{args.model}_{args.aug}.pt')
            
    

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    
    train(args)
    
    
    
    