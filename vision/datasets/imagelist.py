import os.path as osp
from typing import Optional, Callable, List, Tuple, Any

import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as T
from torchvision.datasets.folder import default_loader


class ImageList(tv_datasets.VisionDataset):
    """Image-list dataset: each line is `<relative_or_absolute_path> <label>`."""

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.targets = [s[1] for s in self.samples]
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.loader = default_loader

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        samples = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, target = line.rsplit(maxsplit=1)
                if not osp.isabs(path):
                    path = osp.join(self.root, path)
                samples.append((path, int(target)))
        return samples

    def __getitem__(self, index: int) -> Tuple[Any, int, int]:
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, 0

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return len(self.classes)


class SAMPLE(ImageList):
    image_list = {'S': 'Simulation.txt', 'R': 'Real.txt'}
    CLASSES = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']

    def __init__(self, root: str, task: str, split: str = 'train', **kwargs):
        if task not in self.image_list:
            raise ValueError(f'Unknown SAMPLE task: {task}')
        data_list_file = osp.join(root, self.image_list[task])
        transform = kwargs.pop('transform', None)
        if transform is None:
            if split == 'train':
                transform = T.Compose([
                    T.Resize(128), T.Grayscale(1), T.RandomRotation(45),
                    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor()
                ])
            else:
                transform = T.Compose([T.Resize(128), T.Grayscale(1), T.ToTensor()])
        super().__init__(root, self.CLASSES, data_list_file, transform=transform, **kwargs)


class S2M(ImageList):
    image_list = {'S': 'Simulation.txt', 'R': 'Real.txt'}
    CLASSES = ['2s1', 'bmp2', 'btr70', 't72', 'zsu23']

    def __init__(self, root: str, task: str, split: str = 'train', **kwargs):
        if task not in self.image_list:
            raise ValueError(f'Unknown S2M task: {task}')
        data_list_file = osp.join(root, self.image_list[task])
        transform = kwargs.pop('transform', None)
        if transform is None:
            if split == 'train':
                transform = T.Compose([
                    T.RandomResizedCrop(128, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                    T.Grayscale(1), T.RandomRotation(45),
                    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor()
                ])
            else:
                transform = T.Compose([T.Resize(128), T.Grayscale(1), T.ToTensor()])
        super().__init__(root, self.CLASSES, data_list_file, transform=transform, **kwargs)


class SimulatedSARShip(ImageList):
    image_list = {'S': 'Simulated.txt', 'R': 'Real.txt'}
    CLASSES = ['Carrier', 'Container', 'Tanker']

    def __init__(self, root: str, task: str, split: str = 'train', **kwargs):
        if task not in self.image_list:
            raise ValueError(f'Unknown SimulatedSARShip task: {task}')
        data_list_file = osp.join(root, self.image_list[task])
        transform = kwargs.pop('transform', None)
        if transform is None:
            if split == 'train':
                transform = T.Compose([
                    T.Resize(128), T.Grayscale(1),
                    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor()
                ])
            else:
                transform = T.Compose([T.Resize(128), T.Grayscale(1), T.ToTensor()])
        super().__init__(root, self.CLASSES, data_list_file, transform=transform, **kwargs)
