"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import numpy as np
import os
import os.path as osp
import warnings
from typing import Optional, Callable, Tuple, Any, List, Iterable
import bisect
import copy

import torch
from torch.utils.data.dataset import Dataset, IterableDataset
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import torchvision.transforms as T
from torch.utils.data import Sampler, Subset, ConcatDataset
import torchvision.transforms.functional as F
from PIL import Image

class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.targets = [s[1] for s in self.samples]
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        self.data_list_file = data_list_file

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, torch.tensor(0)

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented


class SAMPLE(ImageList):
    image_list = {"S": "Simulation.txt", "R": "Real.txt", "M1": "MLDG1.txt", "M2": "MLDG2.txt"}
    CLASSES = ['2s1','bmp2','btr70','m1','m2','m35','m60','m548','t72','zsu23']
    
    def __init__(self, root: str, task: str, split: Optional[str] = 'train', **kwargs):
        assert task in self.image_list
        data_list_file = osp.join(root, self.image_list[task])
        super().__init__(root, SAMPLE.CLASSES, data_list_file=data_list_file,** kwargs)
        if self.transform is not None:
            warnings.warn("You have specified transforms")
        else:
            if split == 'train':
                self.transform = T.Compose([
                            T.Resize(128),
                            T.ColorJitter(brightness=0.3, contrast=0.3),
                            T.Grayscale(1),
                            T.RandomRotation(45),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.ToTensor()])
            else:
                self.transform = T.Compose([
                            T.Resize(128),
                            T.Grayscale(1),
                            T.ToTensor()])

    def __getitem__(self, index):
        path, target = self.samples[index]
        domain_label = 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label

    def __len__(self): 
        return len(self.samples)
    
    @property 
    def num_classes(self): 
        return len(self.classes)


class S2M(ImageList):
    image_list = {"S": "Simulation.txt", "R": "Real.txt", "M1": "MLDG1.txt", "M2": "MLDG2.txt"}
    CLASSES = ['2s1', 'bmp2', 'btr70', 't72', 'zsu23']
    
    def __init__(self, root: str, task: str, split: Optional[str] = 'train', **kwargs):
        assert task in self.image_list
        data_list_file = osp.join(root, self.image_list[task])
        super().__init__(root, S2M.CLASSES, data_list_file=data_list_file,** kwargs)
        if self.transform is not None:
            warnings.warn("You have specified transforms")
        else:
            if split == 'train':
                self.transform = T.Compose([
                            T.RandomResizedCrop(size=128, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                            T.ColorJitter(brightness=0.3, contrast=0.3),
                            T.Grayscale(1),
                            T.RandomRotation(45),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.ToTensor()])
            else:
                self.transform = T.Compose([
                            T.Resize(128),
                            T.Grayscale(1),
                            T.ToTensor()])
        print(f"{split}: {self.transform}")

    def __getitem__(self, index):
        path, target = self.samples[index]
        domain_label = 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label

    def __len__(self): 
        return len(self.samples)
    
    @property 
    def num_classes(self): 
        return len(self.classes)



class SimulatedSARShip(ImageList):
    image_list = {"S": "Simulated.txt", "R": "Real.txt", "M1": "MLDG1.txt", "M2": "MLDG2.txt"}
    CLASSES = ['Carrier','Container','Tanker']
    
    def __init__(self, root: str, task: str, split: Optional[str] = 'train', **kwargs):
        assert task in self.image_list
        data_list_file = osp.join(root, self.image_list[task])
        super().__init__(root, SimulatedSARShip.CLASSES, data_list_file=data_list_file,** kwargs)
        if self.transform is not None:
            warnings.warn("You have specified transforms")
        else:
            if split == 'train':
                self.transform = T.Compose([
                            T.Resize(128),
                            # T.ColorJitter(brightness=0.3, contrast=0.3),
                            T.Grayscale(1),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.ToTensor()])
            else:
                self.transform = T.Compose([
                            T.Resize(128),
                            T.Grayscale(1),
                            T.ToTensor()])

    def __getitem__(self, index):
        path, target = self.samples[index]
        domain_label = 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label

    def __len__(self): 
        return len(self.samples)
    
    @property 
    def num_classes(self): 
        return len(self.classes)


# 后面这两个数据集可能有点太难了，域差异太大了。
class FOShip(ImageList):
    # 这个数据集以FuSAR为源域，以OpenSAR为目标域，包含了4个子任务，分别是：O1、O2、O3、O4
    image_list = {"F": "FuSAR.txt", "O1": "OpenSARGRDH_VH.txt", "O2": "OpenSARGRDH_VV.txt", "O3": "OpenSARSLC_VH.txt", "O4": "OpenSARSLC_VV.txt", "M1": "MLDG1.txt", "M2": "MLDG2.txt"}
    CLASSES = ['BulkCarrier','ContainerShip', 'Tanker']
    
    def __init__(self, root: str, task: str, split: Optional[str] = 'train', **kwargs):
        assert task in self.image_list
        data_list_file = osp.join(root, self.image_list[task])
        super().__init__(root, FOShip.CLASSES, data_list_file=data_list_file,** kwargs)
        if self.transform is not None:
            warnings.warn("You have specified transforms")
        else:
            if split == 'train':
                self.transform = T.Compose([
                            T.Resize(128),
                            T.CenterCrop(96),
                            T.RandomResizedCrop(size=96, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
                            T.ColorJitter(brightness=0.3, contrast=0.3),
                            T.Grayscale(1),
                            T.RandomRotation(15),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.ToTensor()])
            else:
                self.transform = T.Compose([
                            T.Resize(128),
                            T.CenterCrop(96),
                            T.Grayscale(1),
                            T.ToTensor()])

    def __getitem__(self, index):
        path, target = self.samples[index]
        domain_label = 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label

    def __len__(self): 
        return len(self.samples)
    
    @property 
    def num_classes(self): 
        return len(self.classes)


class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        # img: PIL Image
        w, h = img.size
        max_side = max(w, h)

        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top

        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)


class OSShip(ImageList):
    # 0: Cargo, 1: Container, 2: Oil Tanker, 3: Warship, 4: Fishing Ship
    image_list = {"O": "Optical.txt", "S": "SAR.txt", "M1": "MLDG1.txt", "M2": "MLDG2.txt"}
    CLASSES = ['0','1','2','3',"4"]
    
    def __init__(self, root: str, task: str, split: Optional[str] = 'train', **kwargs):
        assert task in self.image_list
        data_list_file = osp.join(root, self.image_list[task])
        super().__init__(root, OSShip.CLASSES, data_list_file=data_list_file,** kwargs)
        if self.transform is not None:
            warnings.warn("You have specified transforms")
        else:
            if split == 'train':
                self.transform = T.Compose([
                            PadToSquare(fill=0),
                            T.Resize(128),
                            T.Grayscale(1),
                            T.ColorJitter(brightness=0.3, contrast=0.3),
                            T.RandomRotation(15),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.ToTensor()])
            else:
                self.transform = T.Compose([
                            PadToSquare(fill=0),
                            T.Resize(128),
                            T.Grayscale(1),
                            T.ToTensor()])

    def __getitem__(self, index):
        path, target = self.samples[index]
        domain_label = 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label

    def __len__(self): 
        return len(self.samples)
    
    @property 
    def num_classes(self): 
        return len(self.classes)


class ConcatDatasetWithDomainLabel(ConcatDataset):
    """ConcatDataset with domain label"""

    def __init__(self, *args, **kwargs):
        super(ConcatDatasetWithDomainLabel, self).__init__(*args, **kwargs)
        self.index_to_domain_id = {}
        domain_id = 0
        start = 0
        for end in self.cumulative_sizes:
            for idx in range(start, end):
                self.index_to_domain_id[idx] = domain_id
            start = end
            domain_id += 1

    def __getitem__(self, index):
        img, target, _ = super(ConcatDatasetWithDomainLabel, self).__getitem__(index)
        domain_id = self.index_to_domain_id[index]
        return img, target, domain_id


class RandomDomainSampler(Sampler):
    r"""Randomly sample :math:`N` domains, then randomly select :math:`K` samples in each domain to form a mini-batch of
    size :math:`N\times K`.

    Args:
        data_source (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (:math:`N\times K` here)
        n_domains_per_batch (int): number of domains to select in a single mini-batch (:math:`N` here)
    """

    def __init__(self, data_source: ConcatDataset, batch_size: int, n_domains_per_batch: int):
        super(Sampler, self).__init__()
        self.n_domains_in_dataset = len(data_source.cumulative_sizes)
        self.n_domains_per_batch = n_domains_per_batch
        assert self.n_domains_in_dataset >= self.n_domains_per_batch

        self.sample_idxes_per_domain = []
        start = 0
        for end in data_source.cumulative_sizes:
            idxes = [idx for idx in range(start, end)]
            self.sample_idxes_per_domain.append(idxes)
            start = end

        assert batch_size % n_domains_per_batch == 0
        self.batch_size_per_domain = batch_size // n_domains_per_batch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.n_domains_in_dataset)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = random.sample(domain_idxes, self.n_domains_per_batch)

            for domain in selected_domains:
                sample_idxes = sample_idxes_per_domain[domain]
                if len(sample_idxes) < self.batch_size_per_domain:
                    selected_idxes = np.random.choice(sample_idxes, self.batch_size_per_domain, replace=True)
                else:
                    selected_idxes = random.sample(sample_idxes, self.batch_size_per_domain)
                final_idxes.extend(selected_idxes)

                for idx in selected_idxes:
                    if idx in sample_idxes_per_domain[domain]:
                        sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain:
                    stop_flag = True

        return iter(final_idxes)

    def __len__(self):
        return self.length



# class RandomDomainSampler(Sampler):
#     r"""
#     【3域配对专用采样器】
#     严格保证：每次采样 3 个域中【相同索引的配对图像】
#     适用于：多域图像一一对应的场景（如同一图像的3种风格/3个光源域）
    
#     逻辑：
#     1. 先随机选 配对本地索引 i
#     2. 同时抽取：域1[i]、域2[i]、域3[i]
#     3. 批次由 N 组配对样本组成（总batch_size = 3 × N）
#     """
#     def __init__(self, data_source: ConcatDataset, batch_size: int, n_domains_per_batch=3):
#         """
#         Args:
#             data_source: ConcatDataset 拼接的3个域数据集
#             batch_size: 【总批次大小】（必须是3的倍数，因为3个域配对）
#         """
#         super().__init__(data_source)
#         # 1. 基础校验：必须是3个域
#         self.domains = data_source.datasets
#         self.n_domains = len(self.domains)
#         assert self.n_domains == 3, "当前采样器仅支持3个配对域"

#         # 2. 核心校验：3个域样本数必须完全相同（一一对应）
#         self.n_pairs = len(self.domains[0])  # 配对总数量
#         for domain in self.domains:
#             assert len(domain) == self.n_pairs, "3个域的样本数量必须一一对应！"

#         # 3. 计算每个域的全局索引偏移量
#         # 域0: 0 ~ n_pairs-1
#         # 域1: n_pairs ~ 2n_pairs-1
#         # 域2: 2n_pairs ~ 3n_pairs-1
#         self.offsets = [i * self.n_pairs for i in range(self.n_domains)]

#         # 4. 批次配置：每组配对=3个样本，计算每批次采样多少组配对
#         assert batch_size % 3 == 0, "批次大小必须是3的倍数（3个域配对）"
#         self.pairs_per_batch = batch_size // 3  # 每批次的配对组数
#         self.batch_size = batch_size

#     def __iter__(self):
#         # 1. 随机打乱所有配对索引（无放回采样，保证不重复）
#         pair_indices = list(range(self.n_pairs))
#         random.shuffle(pair_indices)

#         # 2. 按批次分组生成采样索引
#         final_indices = []
#         for i in range(0, len(pair_indices), self.pairs_per_batch):
#             # 取当前批次的配对索引
#             current_pairs = pair_indices[i:i+self.pairs_per_batch]
#             # 为每个配对索引，同时取3个域的全局索引
#             for pair_idx in current_pairs:
#                 for offset in self.offsets:
#                     final_indices.append(offset + pair_idx)

#         # 返回迭代器
#         return iter(final_indices)

#     def __len__(self):
#         # 总采样样本数 = 配对数 × 3
#         return self.n_pairs * 3