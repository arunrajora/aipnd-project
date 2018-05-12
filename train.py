import sys
import time
import argparse
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms, models


def prepare_dataset(data_dir, batch_size, mean, std, im_size):
    '''
        Prepares the dataset for training, validation and testing
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    # Load the datasets with ImageFolder
    image_datasets = {
        'train': ImageFolder(data_dir + '/train', data_transforms['train']),
        'valid': ImageFolder(data_dir + '/valid', data_transforms['valid']),
        'test': ImageFolder(data_dir + '/test', data_transforms['test']),
    }
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size = batch_size, shuffle = True),
        'valid': DataLoader(image_datasets['valid'], batch_size = batch_size, shuffle = True),
        'test': DataLoader(image_datasets['test'], batch_size = batch_size, shuffle = True),
    }
    return image_datasets, dataloaders

def load_model(arch):
    '''
        Loads the pretrained model based on the specified architecture
        Returns the pretrained model
    '''
    if arch == 'vgg19_bn':
        return models.vgg19_bn(pretrained=True)
    elif arch == 'resnet152':
        return models.resnet152(pretrained=True)

def freeze_layers(model):
    '''
        Freezes the layers of the model
        Returns the model
    '''
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_label_count(image_datasets):
    '''
        Returns the number of labels
    '''
    return len(image_datasets['train'].class_to_idx)

def replace_layers(model, arch, hidden_units, image_datasets):
    '''
        Replaces the layers of the model based on the architecture
    '''
    if arch == 'vgg19_bn':
        model.classifier = nn.Sequential(OrderedDict([
            ('Linear1', nn.Linear(25088, hidden_units)),
            ('relu1', nn.ReLU()),
            ('Dropout1', nn.Dropout(p=0.5)),
            ('Linear2', nn.Linear(hidden_units, get_label_count(image_datasets))),
            ('output', nn.LogSoftmax(dim = 1))
        ])) 
    elif arch == 'resnet152':
        model.fc = nn.Sequential(OrderedDict([
            ('Linear1', nn.Linear(2048, hidden_units)),
            ('relu1', nn.ReLU()),
            ('Dropout1', nn.Dropout(p=0.5)),
            ('Linear2', nn.Linear(hidden_units, get_label_count(image_datasets))),
            ('output', nn.LogSoftmax(dim = 1))
        ]))
    return model

def train_model(model, criterion, optimizer, scheduler, epochs, image_datasets, dataloaders, device):
    '''
        Trains the model
    '''
    start_time = time.time()
    # for each epoch
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        
        # run on train and validation batches 
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            # total loss in this epoch
            running_loss = 0.0
            # total number of correct predictions in this epoch
            running_corrects = 0.0
            
            # for each batch
            for inputs, labels in tqdm(dataloaders[phase], desc='{} Batch'.format(phase)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        # backward propogation
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print('For {}:, Loss: {:.4f}, Accuracy: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

    end_time = time.time() - start_time
    print('Training completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
    return model, optimizer

def test_model(model, dataloaders, image_datasets, device):
    '''
        Tests the model
    '''
    model.eval()
    running_corrects = 0
    for inputs, labels in tqdm(dataloaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        running_corrects += torch.sum(predictions == labels.data)
    test_accuracy = running_corrects.double() / len(image_datasets['test'])
    print('Testing Accuracy: {:.4f}%'.format(test_accuracy*100.0))
    

def save_model(trained_model, epochs, optimizer, mean, std, im_size, arch, image_datasets, path_to_save, learning_rate, batch_size):
    '''
        Saves the model
    '''
    trained_model.class_to_idx = image_datasets['train'].class_to_idx
    trained_model.epochs = epochs
    trained_model.optimizer = optimizer.state_dict()
    trained_model.data_params = {
        "mean": mean,
        "std": std,
        "im_size": im_size,
        "arch": arch,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }
    torch.save(trained_model, path_to_save)
    print("Model saved at", path_to_save)

def get_device(enable_gpu):
    '''
        Gets the GPU or CPU device
        Returns torch.device
    '''
    if enable_gpu:
        if torch.cuda.is_available():
            print("(using GPU)\n")
            return torch.device("cuda:0")
        else:
            print("(no GPU found, using CPU)\n")
    else:
        print("(using CPU)\n")
    return torch.device("cpu")

def get_parameters(arch):
    '''
        Gets the parameters for the model based on the architecture
    '''
    # mean
    mean = [0.485, 0.456, 0.406]
    # std
    std = [0.229, 0.224, 0.225]
    # image size
    im_size = 224
    return mean, std, im_size

def main():
    parser = argparse.ArgumentParser(description='Train a model on the data.')
    
    parser.add_argument("--dataset", type=str, help="path to directory containing training/validation/test data", default="data/")
    parser.add_argument('--arch', type=str,choices=["resnet152","vgg19_bn"], help='Architecture of the pretrained model to be used', default="resnet152")
    parser.add_argument("--gpu", help="enable gpu for prediction", action="store_true")
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.0001)
    parser.add_argument("--hidden_units", type=int, help="number of hidden units to be used", default=512)
    parser.add_argument("--epochs", type=int, help="epochs", default=5)
    parser.add_argument("--batch_size", type=int, help="batch size", default=64)
    parser.add_argument("--save_path", type=str, help="location where model is saved", default="checkpoint.model")
    
    args = parser.parse_args()
    # Get the device- GPU or CPU
    device = get_device(args.gpu)
    # Load the model
    model = load_model(args.arch)
    # Load the parameters
    mean, std, im_size = get_parameters(args.arch)
    # Prepare the dataset
    image_datasets, dataloaders = prepare_dataset(args.dataset, args.batch_size, mean, std, im_size)
    
    # Freeze the layers
    model = freeze_layers(model)
    # Replace the layers
    model = replace_layers(model, args.arch, args.hidden_units, image_datasets)
    # Set the model to run on the device
    model = model.to(device)
    
    # Create the optimizer
    optimizer = optim.Adam(model.fc.parameters() if args.arch == 'resnet152' else model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()
    # Create the scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7)
    
    # Start training
    print("\nTraining-\n")
    trained_model, checkpoint_optimizer = train_model(model, criterion, optimizer, scheduler, args.epochs, image_datasets, dataloaders, device)
    
    # Start testing
    print("\nTesting-\n")
    test_model(model, dataloaders, image_datasets, device)
    
    # Save the model
    print("\nSaving Model-\n")
    save_model(model, args.epochs, optimizer, mean, std, im_size, args.arch, image_datasets, args.save_path, args.learning_rate, args.batch_size)

if __name__ == "__main__":
    main()
