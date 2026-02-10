import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy


mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
BATCH_SIZE = 4
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flat_root = 'data/arch_dataset/flat_dataset'

# загрузчик данных в виде тензоров + аугментация
def get_dataloaders():
    chosen_transforms = {'train': transforms.Compose(
        [transforms.RandomResizedCrop(size=256), transforms.RandomRotation(degrees=15),
         transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean_nums, std_nums)]),
                         'val': transforms.Compose(
                             [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize(mean_nums, std_nums)]),
                        'test': transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean_nums, std_nums)
        ])}

    chosen_datasets = {x: datasets.ImageFolder(os.path.join(flat_root, x), chosen_transforms[x]) for x in
                       ['train', 'val', 'test']}

    dataloaders = {
        x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size=4, shuffle=(x == 'train'), num_workers=4) for x in
        ['train', 'val', 'test']}

    dataset_sizes = {x: len(chosen_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = chosen_datasets['train'].classes

    vis_loader = torch.utils.data.DataLoader(
        chosen_datasets['val'],
        batch_size=4,
        shuffle=True,  # только для визуализации
        num_workers=4
    )

    return dataloaders, class_names, dataset_sizes, vis_loader

# Visualize some images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean_nums)
    std = np.array(std_nums)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)

# обучение для всех моделей, без тестовой выборки ?? (мб можно рассмотреть и обучение всех архитектур с тестовой, но потом определимся)
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train() # Set model to training mode
            else: model.eval() # Set model to evaluate mode

            current_loss = 0.0
            current_corrects = 0

            print('Iterating through data...')

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Time to carry out the forward training poss
                # We only need to log the loss stats if we are in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} | Acc: {:.4f}'.format( phase, epoch_loss, epoch_acc))

            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_since // 60, time_since % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, vis_loader, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_handeled = 0
    fig = plt.figure(figsize=(6, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(vis_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_handeled += 1
                ax = plt.subplot(num_images//2, 2, images_handeled)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}\nreal: {class_names[labels[j]]}', fontsize=10)
                imshow(inputs.cpu().data[j])
                if images_handeled == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)