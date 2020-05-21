import json
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
from collections import OrderedDict
import seaborn as sns
import time

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_data, batch_size=32)
valid_dataloaders = torch.utils.data.DataLoader(valid_data, batch_size=32)


def getDevice(gpu=False):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    return device


def validation(model, valid_loader, criterion, gpu=False):
    valid_loss = 0
    accuracy = 0
    device = getDevice(gpu)
    # change model to work with either cuda or cpu
    model.to(device)

    # load images and labels from iterated valid loader
    for i, (images, labels) in enumerate(valid_loader):
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)

        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)

        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


def get_pre_trained_model(arch='vgg16'):
    model = ""
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    return model


def network(arch='alexnet', lr=0.001, hidden_units=512):
    architectures = {"vgg16": 25088, "densenet121": 1024, "alexnet": 9216}
    if arch not in architectures:
        print('Sorry you can select the following achitectures vgg16,densenet121 or alexnet')
        return
    # lr = 0.0008
    # lr = 0.001
    model = get_pre_trained_model(arch)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(architectures[arch], hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.05)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    return model, criterion, optimizer


def save_checkpoint(arch='alexnet'):
    model, criterion, optimizer = network(arch)
    model.class_to_idx = train_data.class_to_idx
    torch.save({'structure': arch,
                'hidden_layer': 512,
                'droupout': 0.5,
                'epochs': 3,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'optimizer_dict': optimizer.state_dict()},
               'checkpoint.pth')
    return True


def test_network(testloader, model, gpu):
    device = getDevice(gpu)
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nNetwork accuracy on test network : %d %%' % (100 * correct / total))


def train_network(model, optimizer, criterion, gpu, arch, epochs=3):
    device = getDevice(gpu)
    # epochs = 3
    steps = 0
    print_every = 40

    # change to gpu mode
    model.to(device)
    print('Training started ...')
    for e in range(epochs):
        since = time.time()
        running_loss = 0

        # Iterating over data to carry out training step
        for i, (images, labels) in enumerate(train_dataloaders):
            steps += 1

            images, labels = images.to(device), labels.to(device)

            # Clear the gradients from all Variables
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Carrying out validation step
            if steps % print_every == 0:
                # Set model to eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_dataloaders, criterion)

                print("Epoch: {}/{} | ".format(e + 1, epochs),
                      "Training Loss: {:.2f} | ".format(running_loss / print_every),
                      "Validation Loss: {:.2f} | ".format(valid_loss / len(valid_dataloaders)),
                      "Validation Accuracy: {:.2f}%".format(accuracy / len(valid_dataloaders) * 100))

                running_loss = 0

                # Turning training back on
                model.train()

    print('Training ended ...')
    model.class_to_idx = train_data.class_to_idx
    # save trained model
    torch.save({'structure': arch,
                'hidden_layer': 512,
                'droupout': 0.5,
                'epochs': 3,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'optimizer_dict': optimizer.state_dict()},
               'checkpoint.pth')

    test_network(test_dataloaders, model, gpu)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    model, _, _ = network(structure)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    pil_image = Image.open(image)

    process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    image_tensor = process(pil_image)
    # np_image = np.array(pil_image)
    return image_tensor


def predict_image(model, image_path, device, topk):
    if torch.cuda.is_available():
        model.to(device)
    image = process_image(image_path)
    model.eval()
    img = image.unsqueeze_(0)
    with torch.no_grad():
        output = model.forward(img)

    probs = torch.exp(output)
    k_prob, k_index = probs.topk(topk)[0], probs.topk(topk)[1]

    # convert to  list
    probs = k_prob.numpy()[0]
    k_index_list = k_index.numpy()[0]

    indx_to_class = {model.class_to_idx[i]: i for i in model.class_to_idx}

    classes = list()
    [classes.append(indx_to_class[index]) for index in k_index_list]

    return probs, classes


def get_cat_to_name(cat_json):
    with open(cat_json, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name


def get_flower_name(model, image_path, gpu, cat_to_name, topk):
    device = getDevice(gpu)
    probs, classes = predict_image(model, image_path, device, topk)
    print("\n Below are Probabilities from predicted image")
    print(probs)
    print("\n Below are Probability classes from predicted image")
    print(classes)
    max_index = np.argmax(probs)
    title_from_class = classes[max_index]
    cat_to_name = get_cat_to_name(cat_to_name)

    probable_names = list()

    {probable_names.append(cat_to_name[str(i)]) for i in classes}

    flower_name = cat_to_name[str(title_from_class)]

    return flower_name, probable_names
