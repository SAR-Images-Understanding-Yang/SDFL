import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import vision.datasets as datasets
import vision.models as models
from utils.metric import accuracy, ConfusionMatrix
from utils.meter import AverageMeter, ProgressMeter


class ForeverDataIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = iter(data_loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            return next(self.iterator)


def get_model(model_name, pretrained=True):
    if model_name not in models.__dict__:
        raise ValueError(f'Unsupported backbone: {model_name}')
    return models.__dict__[model_name](pretrained=pretrained)


def get_dataloaders(args, train_tasks, test_task, train_transform=None, val_transform=None):
    dataset_map = {
        'SAMPLE': datasets.SAMPLE,
        'S2M': datasets.S2M,
        'SimulatedSARShip': datasets.SimulatedSARShip,
    }
    if args.data not in dataset_map:
        raise ValueError(f'Unsupported dataset: {args.data}')
    if not train_tasks or len(train_tasks) != 1:
        raise ValueError('The released SDFL protocol expects exactly one synthetic source task.')
    if not test_task or len(test_task) != 1:
        raise ValueError('The released SDFL protocol expects exactly one measured target task.')

    dataset_class = dataset_map[args.data]
    train_dataset = dataset_class(root=args.root, task=train_tasks[0], split='train', transform=train_transform)
    val_dataset = dataset_class(root=args.root, task=train_tasks[0], split='val', transform=val_transform)
    test_dataset = dataset_class(root=args.root, task=test_task[0], split='test', transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    args.class_names = test_dataset.CLASSES
    return ForeverDataIterator(train_loader), val_loader, test_loader, test_dataset.num_classes


def validate(data_loader, model, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(data_loader), [batch_time, losses, top1], prefix='Test: ')

    model.eval()
    confmat = ConfusionMatrix(len(args.class_names))
    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(data_loader):
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss = F.cross_entropy(output, target)
            acc1 = accuracy(output, target)[0]
            confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
    print(f' * Acc@1 {top1.avg:.3f}')
    return top1.avg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
