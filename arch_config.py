import pandas as pd
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from sklearn.metrics import f1_score, top_k_accuracy_score, balanced_accuracy_score, confusion_matrix
import seaborn as sns


mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flat_data = '/data/arch_dataset/flat_dataset'

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1 = float('-inf')
        self.early_stop = False

    def __call__(self, val_f1):
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.counter = 0
            print(f"🌟 New best F1_macro: {val_f1:.4f}")
        else:
            self.counter += 1
            print(f"⏳️ No improvements: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 Early stop! patience={self.patience}")
        return self.early_stop

RESIZE_SIZE = 384  # rn18: 256, enb4: 384
IMG_SIZE = 380     # rn18: 224, enb4: 380

# загрузчик данных в виде тензоров + аугментация
def get_dataloaders():
    chosen_transforms = {
        'train': transforms.Compose([
            # 1. КРОШКА И РОТАЦИЯ (главное для архитектуры!)
            transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.TrivialAugmentWide(),

            # 2. ГЕОМЕТРИЯ
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

            # 3. ЦВЕТ (важно для фото архитектуры!)
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),

            # 4. ТЕНЗОР + НОРМАЛИЗАЦИЯ
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
        ]),

        'val': transforms.Compose([
            transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC), # 256 rn18
            transforms.CenterCrop(IMG_SIZE),  # 224x224 rn18, 380 enb4
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
        ]),

        'test': transforms.Compose([
            transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
        ])
    }

    chosen_datasets = {x: datasets.ImageFolder(os.path.join(flat_data, x), chosen_transforms[x]) for x in
                       ['train', 'val', 'test']}

    dataloaders = {
        x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=4) for x in
        ['train', 'val', 'test']}

    dataset_sizes = {x: len(chosen_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = chosen_datasets['train'].classes

    vis_loader = torch.utils.data.DataLoader(
        chosen_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,  # только для визуализации
        num_workers=4
    )

    return dataloaders, class_names, dataset_sizes, vis_loader

def get_dataloaders50():
    chosen_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=384, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            # TrivialAugmentWide сам сделает повороты и игры с цветом
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
        ]),
        'val': transforms.Compose([
            transforms.Resize(410),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
        ]),
        'test': transforms.Compose([
            transforms.Resize(410),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
        ])
    }

    chosen_datasets = {x: datasets.ImageFolder(os.path.join(flat_data, x), chosen_transforms[x]) for x in
                       ['train', 'val', 'test']}

    dataloaders = {
        x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=4) for x in
        ['train', 'val', 'test']}

    dataset_sizes = {x: len(chosen_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = chosen_datasets['train'].classes

    vis_loader = torch.utils.data.DataLoader(
        chosen_datasets['val'],
        batch_size=BATCH_SIZE,
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
    best_wacc = 0.0
    best_top3 = 0.0
    best_top1 = 0.0
    best_f1 = 0.0

    wacc_train = []
    wacc_val = []

    f1_train = []
    f1_val = []

    top3_train = []
    top3_val = []

    top1_train = []
    top1_val = []

    loss_train = []
    loss_val = []

    early_stopper = EarlyStopping(patience=10, min_delta=0.0005)

    for epoch in range(num_epochs):
        epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            all_preds = []
            all_labels = []
            all_probs = []

            if phase == 'train':
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
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.detach().cpu().numpy())

            epoch_f1 = f1_score(all_labels, all_preds, average='macro')
            epoch_top3 = top_k_accuracy_score(all_labels, np.vstack(all_probs), k=3)
            epoch_top1 = top_k_accuracy_score(all_labels, np.vstack(all_probs), k=1)
            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_wacc = balanced_accuracy_score(all_labels, all_preds, adjusted=True)
            epoch_acc = current_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} | Acc: {:.4f} | Weighted Acc: {:.4f} | F1-macro: {:.4f} | Top-1 Acc: {:.4f} | Top-3 Acc: {:.4f}'.format( phase, epoch_loss, epoch_acc, epoch_wacc, epoch_f1, epoch_top1, epoch_top3))

            if phase == 'train':
                scheduler.step()
                wacc_train.append(epoch_wacc)
                f1_train.append(epoch_f1)
                top3_train.append(epoch_top3)
                top1_train.append(epoch_top1)
                loss_train.append(epoch_loss)
            else:
                wacc_val.append(epoch_wacc)
                f1_val.append(epoch_f1)
                top3_val.append(epoch_top3)
                top1_val.append(epoch_top1)
                loss_val.append(epoch_loss)
                early_stopper(epoch_f1)

                if epoch_f1 > best_f1 or (epoch_top1 > best_top1 and epoch_f1 >= best_f1):
                    best_f1 = epoch_f1
                    best_top1 = epoch_top1 if epoch_top1 > best_top1 else best_top1
                    best_top3 = epoch_top3 if epoch_top3 > best_top3 else best_top3
                    best_wacc = epoch_wacc if epoch_wacc > best_wacc else best_wacc
                    best_model_wts = copy.deepcopy(model.state_dict())

            del all_preds, all_labels, all_probs
        print('Epoch Time: {:.4f}s'.format(time.time() - epoch_time))
        print()

        if early_stopper.early_stop and epoch + 1 != num_epochs:
            print(f"✅ Early stopping on epoch {epoch + 1}")
            break

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_since // 60, time_since % 60))
    print('Best val Weighted Acc: {:4f} | Best F1-score: {:.4f} | Best Top-1 Acc: {:.4f} | Best Top-3 Acc: {:.4f}'.format(best_wacc, best_f1, best_top1, best_top3))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    return model, (wacc_train, f1_train, top3_train, top1_train, loss_train, wacc_val, f1_val, top3_val, top1_val, loss_val)

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

def visualize_metrics(metrics, title, epochs=25):
    x = list(range(1, epochs + 1))
    accuracy_train, f1_train, top3_train, top1_train, loss_train, accuracy_val, f1_val, top3_val, top1_val, loss_val = metrics

    train = [accuracy_train, f1_train, top3_train, top1_train, loss_train, top3_val]
    val = [accuracy_val, f1_val, top3_val, top1_val, loss_val, top1_val]
    titles = ["Weighted accuracy", "F1-score macro", "Top-3 Accuracy", "Top-1 Accuracy", "Loss function",
              "Top-1 vs Top-3 Validation Accuracy"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=20)

    k = 0
    for i in range(2):
        for j in range(3):
            t, v, plot_name = train[k], val[k], titles[k]
            if k != 5:
                sns.lineplot(x=x, y=t, ax=axes[i, j], label='Train', legend='auto')
                sns.lineplot(x=x, y=v, ax=axes[i, j], label='Val', legend='auto')
            else:
                sns.lineplot(x=x, y=t, ax=axes[i, j], label='Top-3', legend='auto')
                sns.lineplot(x=x, y=v, ax=axes[i, j], label='Top-1', legend='auto')
            axes[i, j].set_title(plot_name)
            axes[i, j].set_xlabel("Epoch")
            axes[i, j].set_ylabel("Metric value")
            axes[i, j].grid(True, alpha=0.5)
            k += 1

    plt.tight_layout()
    plt.show()

def visualize_cm(model, class_names, test_set):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_set:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=.5)

    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix', fontsize=20, fontweight='bold')
    plt.show()