# Author: Yash P. Pawar
# Flower Classifier

import logging
import argparse
import sys
from pathlib import Path
from workspace_utils import active_session

import numpy as np
from torchvision import models, datasets, transforms
import torch
from torch import nn, optim

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
BATCH_SIZE = 64

# archs = list(filter(lambda x: (not x.startswith('_')) and ('Weights' not in x) , dir(models)))
archs = ['vgg16', 'vgg11', 'resnet50']

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=Path)
parser.add_argument('--save-dir', default='.', type=Path)
parser.add_argument('-a', '--arch', choices=archs, default='resnet50')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('-lr', '--learning-rate', default=4e-3, type=float)
parser.add_argument('--hidden-units', default=512, type=int)
parser.add_argument('-e', '--epochs', default=5, type=int)

args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(args.save_dir / "train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"Data Dir: {args.data_dir}")
logging.info(f"Architecture: {args.arch}")
logging.info(f"Epochs: {args.epochs}")
logging.info(f"GPU: {args.gpu}")
logging.info(f"Number Of Hidden Inputs: {args.hidden_units}")
logging.info(f"Learning Rate: {args.learning_rate}\n\n")

logging.info("Loading the Data and creating batches")
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

trainset = datasets.ImageFolder(args.data_dir / 'train', transform=train_transforms)
testset  = datasets.ImageFolder(args.data_dir / 'test', transform=test_transforms)
validset = datasets.ImageFolder(args.data_dir / 'valid', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
validloader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE)

logging.info("Data Loaded")
logging.info("Creating The Model")

Model = getattr(models, args.arch)

model = Model(pretrained=True)

# freeze the model
for param in model.parameters():
    param.requires_grad = False

if 'vgg' in args.arch:
    input_features = model.classifier[0].in_features
elif 'resnet' in args.arch:
    input_features = model.fc.in_features

output_features = len(trainset.classes)

classifier = nn.Sequential(
    nn.Linear(input_features, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(args.hidden_units, output_features),
    nn.LogSoftmax(dim=1)
)

if 'vgg' in args.arch:
    model.classifier = classifier
elif 'resnet' in args.arch:
    model.fc = classifier

device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")

logging.info(f"Using {device} Acceleration")

if 'vgg' in args.arch:
    parameters = model.classifier.parameters()
elif 'resnet' in args.arch:
    parameters = model.fc.parameters()

criterion = nn.NLLLoss()
optimizer = optim.Adam(parameters, lr=args.learning_rate)
model.to(device)

logging.info("Training The Model")

with active_session():
    try: 
        for epoch in range(args.epochs):
            running_loss = 0
            steps = 0

            for X_train, y_train in trainloader:
                steps += 1
                X_train = X_train.to(device)
                y_train = y_train.to(device)

                optimizer.zero_grad() 

                log_preds = model(X_train)
                loss = criterion(log_preds, y_train)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                print(f"[{steps}-{epoch+1}/{args.epochs}] Training Loss: {running_loss/steps:.3f}")
            else:
                valid_loss, accuracy = 0, 0
                model.eval()
                n_valids = len(validloader)

                with torch.no_grad():
                    for X_valid, y_valid in validloader:
                        X_valid = X_valid.to(device)
                        y_valid = y_valid.to(device)

                        log_preds = model(X_valid)
                        batch_loss = criterion(log_preds, y_valid)

                        valid_loss += batch_loss.item()

                        preds = torch.exp(log_preds)
                        _, top_class = preds.topk(1, dim=1)
                        equals = top_class == y_valid.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                logging.info(f"[{epoch+1}/{args.epochs}] Training Loss: {running_loss/n_valids:.3f} | Valid Loss: {valid_loss/n_valids:.3f}, Accuracy: {accuracy/n_valids:.3f}%")   

                model.train()

        logging.info("Done Training")
    except KeyboardInterrupt:
        logging.info("Stopping Training")

logging.info("Starting Test")

test_loss = 0
test_accuracy = 0

model.eval()

with torch.no_grad():
    for X_test, y_test in testloader:
        X_test, y_test = X_test.to(device), y_test.to(device) 

        log_preds = model(X_test)
        batch_loss = criterion(log_preds, y_test)

        test_loss += batch_loss.item()

        ps = torch.exp(log_preds)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y_test.view(*top_class.shape)

        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

logging.info(f"Test Loss: {test_loss/len(testloader):.3f}")
logging.info(f"Test Accuracy: {test_accuracy/len(testloader):.3f}")

logging.info(f"Saving Checkpoint at path: {args.save_dir / 'checkpoint.pth'}")

checkpoint = {
    "state_dict": model.state_dict(),
    "class_to_idx": trainset.class_to_idx,
    "input_units": input_features,
    "hidden_units": args.hidden_units,
    "output_units": output_features,
    "arch": args.arch,
    "device": str(device),
}

torch.save(checkpoint, args.save_dir / 'checkpoint.pth')

logging.info("Done!")
