#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :visualize.py
# @Time      :2024/6/11 20:16
# @Author    :Yangxinpeng
# @Introduce :
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn import metrics
import os
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
from matplotlib.colors import to_rgb, hsv_to_rgb, rgb_to_hsv
from sklearn.metrics import f1_score, recall_score, precision_score



# 可视化部分
class Visualize(object):
    '''
    Visualize features by TSNE
    '''
    def __init__(self, root):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.save_path = os.path.join(root, "visualize")
        
        # self.log = None
        # for file in os.listdir(root):
        #     if file.startswith("train"):
        #         self.log = file
        #         continue
        # # 解析log文件
        # # load loss and accuracy
        # self.loss = []
        # self.acc = []
        # with open(self.log, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         if 'Epoch' in line:
        #             for element in line.split(','):
        #                 name, data = element.split(':')
        #                 if name.replace(' ', '') == 'total_loss':
        #                     self.loss.append(float(data))
        #                 elif name.replace(' ', '') == 'test_acc':
        #                     self.acc.append(float(data))


    def adjust_color_brightness(self, rgb_color, brightness_factor):
        """
        调整RGB颜色的亮度（深浅）
        参数：
            rgb_color: 基础颜色，如'#E67E22'或(r, g, b)
            brightness_factor: 亮度因子，>1变亮，<1变暗（建议0.4~1.2）
        返回：
            调整后的十六进制颜色字符串
        """
        # 转换为RGB数组（0~1）
        rgb = np.array(to_rgb(rgb_color))
        # 转换为HSV
        hsv = rgb_to_hsv(rgb)
        # 调整亮度通道（V），限制在0~1之间
        hsv[2] = np.clip(hsv[2] * brightness_factor, 0, 1)
        # 转回RGB
        new_rgb = hsv_to_rgb(hsv)
        # 转换为十六进制
        return '#{:02x}{:02x}{:02x}'.format(
            int(new_rgb[0]*255), int(new_rgb[1]*255), int(new_rgb[2]*255)
        )

    # def plot_tsne(self, features, labels, domains, save_eps=False, 
    #               domain_markers=None, class_names=None, color_list=None):
    #     '''
    #     绘制多域数据的TSNE图，不同类别用不同基础色，不同域用同色不同深浅（+形状辅助）
    #     参数：
    #         features: 特征数据，形状为 (n_samples, n_features)
    #         labels: 类别标签，形状为 (n_samples,)
    #         domains: 域标签，形状为 (n_samples,)
    #         save_eps: 是否保存eps和png文件，默认False
    #         domain_markers: 域对应的形状，字典类型，如 {0: 'o', 1: 's'}，默认提供常用形状
    #         class_names: 类别名称列表，默认是['2s1', 'bmp2', ..., 'zsu23']
    #         color_list: 类别对应的基础颜色列表，默认使用优化后的高对比度颜色
    #     '''
    #     # 设置默认的域形状（辅助区分）
    #     if domain_markers is None:
    #         domain_markers = {0: 'o', 1: '*', 2: '^', 3: 'D', 4: 'v',
    #                           5: '<', 6: '>', 7: 'p', 8: 's', 9: 'h'}
    #     # 设置默认的类别名称
    #     if class_names is None:
    #         class_names = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
    #     # 设置默认的高对比度基础颜色列表
    #     if color_list is None:
    #         color_list = [
    #             '#E67E22', '#7F8C8D', '#2980B9', '#3498DB', '#27AE60',
    #             '#2ECC71', '#8E44AD', '#E91E63', '#8B4513', '#C0392B'
    #         ]
        
    #     # 1. t-SNE降维+归一化
    #     tsne = TSNE(n_components=2, init='pca', random_state=0)
    #     features_tsne = tsne.fit_transform(features)
    #     scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #     result = scaler.fit_transform(features_tsne)

    #     # 2. 画布设置（放大+字体优化）
    #     plt.figure(figsize=(10, 8))
    #     plt.title('t-SNE Visualization of Multi-Domain Data (Color Shades for Domains)', fontsize=14)
    #     plt.xlabel('t-SNE Dimension 1', fontsize=12)
    #     plt.ylabel('t-SNE Dimension 2', fontsize=12)

    #     # 获取唯一的域和类别，并为每个域分配亮度因子
    #     unique_domains = np.unique(domains)
    #     unique_labels = np.unique(labels)
    #     # 亮度因子列表：域0（最亮）→域n（逐渐变暗），可根据需要调整
    #     brightness_factors = {domain: 1.0 - (i * 0.2) for i, domain in enumerate(unique_domains)}
    #     # 限制亮度因子在合理范围（0.4~1.0）
    #     for domain in brightness_factors:
    #         brightness_factors[domain] = max(0.4, brightness_factors[domain])

    #     # 用于存储图例元素
    #     legend_elements = []
    #     legend_labels = []

    #     # 3. 遍历绘制：类别→基础色，域→深浅色+形状
    #     for label in unique_labels:
    #         # 获取当前类别的基础颜色
    #         base_color = color_list[label]
    #         for domain in unique_domains:
    #             # 1. 获取当前类别+当前域的样本索引
    #             domain_idx = np.where(domains == domain)[0]
    #             label_in_domain = labels[domain_idx] == label
    #             idx = domain_idx[np.where(label_in_domain)[0]]
    #             if len(idx) == 0:
    #                 continue

    #             # 2. 生成当前域对应的深浅色
    #             shade_color = self.adjust_color_brightness(base_color, brightness_factors[domain])
    #             # 3. 获取当前域的形状
    #             marker = domain_markers.get(domain, 'o')

    #             # 4. 绘制散点：深浅色+形状，加白色描边提升清晰度
    #             plt.scatter(
    #                 result[idx, 0], result[idx, 1],
    #                 c=shade_color, marker=marker,
    #                 s=40, alpha=0.8, edgecolors='w', linewidths=0.5
    #             )

    #             # 5. 添加图例（类别+域的组合，避免重复）
    #             legend_key = f'{class_names[label]} (Domain {domain})'
    #             if legend_key not in legend_labels:
    #                 legend_elements.append(plt.Line2D(
    #                     [0], [0], marker=marker, color='w', 
    #                     markerfacecolor=shade_color, markersize=10,
    #                     label=legend_key
    #                 ))
    #                 legend_labels.append(legend_key)

    #     # 4. 图例设置：移到图外，避免遮挡
    #     plt.legend(
    #         handles=legend_elements, 
    #         loc='upper left',
    #         bbox_to_anchor=(1.02, 1),
    #         fontsize=9,
    #         framealpha=1
    #     )
    #     # 预留图例空间，调整布局
    #     plt.tight_layout(rect=[0, 0, 0.85, 1])

    #     # 5. 保存图片
    #     if save_eps:
    #         os.makedirs(self.save_path, exist_ok=True)
    #         plt.savefig(os.path.join(self.save_path, 'tsne_multi_domain_shades.png'), dpi=600, format='png', bbox_inches='tight')
    #         plt.savefig(os.path.join(self.save_path, 'tsne_multi_domain_shades.eps'), dpi=600, format='eps', bbox_inches='tight')

    #     # plt.show()
    #     plt.close()

    def plot_tsne(self, features, labels, domains, save_eps=False, 
                  domain_markers=None, class_names=None, color_list=None):
        '''
        绘制多域数据的TSNE图：修复图例杂乱问题
        '''
        # ========== 原有参数逻辑（不变） ==========
        if domain_markers is None:
            domain_markers = {0: 'o', 1: '*', 2: '^', 3: 'D', 4: 'v',
                              5: '<', 6: '>', 7: 'p', 8: 's', 9: 'h'}
        if class_names is None:
            class_names = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
        if color_list is None:
            color_list = [
                '#E67E22', '#2980B9', '#27AE60', '#8E44AD', '#E91E63',
                '#C0392B', '#1ABC9C', '#F39C12', '#9B59B6', '#34495E'
            ]
        
        # ========== 原有降维+归一化（不变） ==========
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features_tsne = tsne.fit_transform(features)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        result = scaler.fit_transform(features_tsne)

        # ========== 原有画布设置（不变） ==========
        plt.figure(figsize=(10, 8))
        plt.title('t-SNE: Class (Shape) & Domain (Color)', fontsize=14)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)

        # ========== 原有标签提取+颜色映射（不变） ==========
        unique_domains = np.unique(domains)
        unique_labels = np.unique(labels)
        domain_color_map = {domain: color_list[i % len(color_list)] for i, domain in enumerate(unique_domains)}

        # ========== 【核心修改：拆分图例为“类别（形状）”和“域（颜色）”】 ==========
        # 1. 单独存储“类别-形状”图例（每个类别只加一次）
        class_legend = []
        class_legend_keys = []
        # 2. 单独存储“域-颜色”图例（每个域只加一次）
        domain_legend = []
        domain_legend_keys = []

        for label in unique_labels:
            class_marker = domain_markers.get(label, 'o')
            # 先加“类别-形状”图例（仅一次）
            class_key = f'Class: {class_names[label]}'
            if class_key not in class_legend_keys:
                class_legend.append(plt.Line2D(
                    [0], [0], marker=class_marker, color='w',
                    markerfacecolor='gray', markersize=10, label=class_key
                ))
                class_legend_keys.append(class_key)

            for domain in unique_domains:
                # 原有索引筛选（不变）
                domain_idx = np.where(domains == domain)[0]
                label_in_domain = labels[domain_idx] == label
                idx = domain_idx[np.where(label_in_domain)[0]]
                if len(idx) == 0:
                    continue

                domain_color = domain_color_map[domain]
                # 原有散点绘制（不变）
                plt.scatter(
                    result[idx, 0], result[idx, 1],
                    c=domain_color, marker=class_marker,
                    s=40, alpha=0.8, edgecolors='w', linewidths=0.5
                )

                # 再加“域-颜色”图例（仅一次）
                domain_key = f'Domain: {domain}'
                if domain_key not in domain_legend_keys:
                    domain_legend.append(plt.Line2D(
                        [0], [0], marker='o', color='w',
                        markerfacecolor=domain_color, markersize=10, label=domain_key
                    ))
                    domain_legend_keys.append(domain_key)

        # ========== 【合并图例，加分隔线区分】 ==========
        all_legend = class_legend + [plt.Line2D([0], [0], linestyle='--', color='gray', label='---')] + domain_legend
        plt.legend(
            handles=all_legend, 
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=10,  # 调整字体大小适配
            framealpha=1,
            ncol=1
        )

        # ========== 原有布局+保存（不变） ==========
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if save_eps:
            os.makedirs(self.save_path, exist_ok=True)
            plt.savefig(os.path.join(self.save_path, 'tsne_multi_domain.png'), dpi=600, format='png', bbox_inches='tight')
            plt.savefig(os.path.join(self.save_path, 'tsne_multi_domain.eps'), dpi=600, format='eps', bbox_inches='tight')

        plt.close()

    def plot_confusion_matrix(self, labels, preds, title=None, thresh=0.8, axis_labels=None, save_eps=False):
        # 利用sklearn中的函数生成混淆矩阵并归一化
        labels_name = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
        cm = metrics.confusion_matrix(labels, preds, labels=None, sample_weight=None)  # 生成混淆矩阵
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
        plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.colorbar()  # 绘制图例
        # 图像标题
        if title is not None:
            plt.title(title)
        # 绘制坐标
        num_local = np.array(range(len(labels_name)))
        if axis_labels is None:
            axis_labels = labels_name
        plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
        plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
        for i in range(np.shape(cm)[0]):
            for j in range(np.shape(cm)[1]):
                if int(cm[i][j] * 100 + 0.5) > 0:
                    plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                            ha="center", va="center",
                            color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
        if save_eps:
            plt.savefig(os.path.join(self.save_path, 'ConfusionMatrix.png'), dpi=600, format='png')
            plt.savefig(os.path.join(self.save_path, 'ConfusionMatrix.eps'), dpi=600, format='eps')

        # plt.show()
        plt.close()

    def plot_loss_curve(self, save_eps=False):
        # 绘制图表
        if self.log is None:
            return
        else:
            # plt.figure(figsize=(10, 6))  # 设置图表大小
            # 绘制端到端(X)的折线图
            plt.plot(range(len(self.loss)), self.loss, label='Loss', marker='o')
            # 设置X轴和Y轴的标签
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # 设置图表的标题
            plt.title('Loss Curve')
            # 添加图例
            plt.legend(framealpha=1)

            # 显示图表
            plt.grid(True)  # 添加网格线
            if save_eps:
                plt.savefig(os.path.join(self.save_path, 'LossCurve.png'), dpi=600, format='png')
                plt.savefig(os.path.join(self.save_path, 'LossCurve.eps'), dpi=600, format='eps')

            plt.show()

    def plot_accuracy_curve(self, save_eps=False):
        # 绘制图表
        if self.log is None:
            return
        else:
            # plt.figure(figsize=(10, 6))  # 设置图表大小
            # 绘制端到端(X)的折线图
            plt.plot(range(len(self.acc)), self.acc, label='Accuracy', marker='s')
            # 设置X轴和Y轴的标签
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # 设置图表的标题
            plt.title('Accuracy Curve')
            # 添加图例
            plt.legend(framealpha=1)
            # 显示图表
            plt.grid(True)  # 添加网格线
            if save_eps:
                plt.savefig(os.path.join(self.save_path, 'AccuracyCurve.png'), dpi=600, format='png')
                plt.savefig(os.path.join(self.save_path, 'AccuracyCurve.eps'), dpi=600, format='eps')

            plt.show()

    def collect_feature(self, 
                        data_loader, 
                        feature_extractor: nn.Module, 
                        classifier: nn.Module,
                        device: torch.device,
                        max_num_features=None) -> torch.Tensor:
        """
        Fetch data from `data_loader`, and then use `feature_extractor` to collect features. This function is
        specific for domain generalization because each element in data_loader is a tuple
        (images, labels, domain_labels).

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader.
            feature_extractor (torch.nn.Module): A feature extractor.
            device (torch.device)
            max_num_features (int): The max number of features to return

        Returns:
            Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
        """
        all_features = []
        all_targets = []
        all_domain_labels = []
        all_preds = []
        correct = 0
        len_target_dataset = len(data_loader.dataset)
        loop = tqdm(data_loader)
        with torch.no_grad():
            for i, (images, target, domain_labels) in enumerate(loop):
                if type(images) == list:
                    images = torch.cat(images, dim=0)
                    target = torch.cat(target, dim=0)
                    domain_labels = torch.cat(domain_labels, dim=0)

                if max_num_features is not None and i >= max_num_features:
                    break
                images = images.to(device)
                feature = feature_extractor(images)
                pred = classifier(images)
                all_features.append(feature.cpu())
                all_targets.append(target.cpu())
                all_domain_labels.append(domain_labels)
                all_preds.append(torch.max(pred, 1)[1])
                correct += torch.sum(torch.max(pred, 1)[1].cpu() == target)
                loop.set_description(f'Model Testing -> ')
                info_dict = {'acc': 100. * correct / len_target_dataset}
                loop.set_postfix(info_dict)

        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_domain_labels = torch.cat(all_domain_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        return (all_features, all_targets, all_domain_labels, all_preds)

    def visualize(self, classifier, train_data_loader, test_data_loader):
        classifier.eval()
        classifier.training = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feature_extractor =  nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        # train data loader
        train_all_features, train_all_targets, train_all_domain_labels, train_all_preds = self.collect_feature(train_data_loader, 
                                                                                                               feature_extractor, 
                                                                                                               classifier, 
                                                                                                               device=device)
        # 这里还是不对，train 和test loader的域都是从0开始的
        max_domain_id = torch.max(train_all_domain_labels)
        # test data loader
        test_all_features, test_all_targets, test_all_domain_labels, test_all_preds = self.collect_feature(test_data_loader, 
                                                                                                           feature_extractor, 
                                                                                                           classifier, 
                                                                                                           device=device)
        test_all_domain_labels += (max_domain_id+1)

        all_features = torch.cat([train_all_features, test_all_features], dim=0)
        all_labels = torch.cat([train_all_targets, test_all_targets], dim=0)
        all_domain_labels = torch.cat([train_all_domain_labels, test_all_domain_labels], dim=0)
        self.plot_tsne(all_features, all_labels, all_domain_labels, save_eps=True)
        self.plot_confusion_matrix(test_all_targets.cpu().numpy(), test_all_preds.cpu().numpy(), save_eps=True)
        
        # self.plot_loss_curve(save_eps=True)
        # self.plot_accuracy_curve(save_eps=True)

        # 计算recall, precision, f1
        macro_recall = recall_score(test_all_targets, test_all_preds.cpu(), average='macro')  # 宏平均召回率
        macro_precision = precision_score(test_all_targets, test_all_preds.cpu(), average='macro')  # 宏平均精确度
        macro_f1 = f1_score(test_all_targets, test_all_preds.cpu(), average='macro')

        print(f'Macro Precision: {macro_precision}')
        print(f'Macro Recall: {macro_recall}')
        print(f'Macro F1 Score: {macro_f1}')
        classifier.train()
        classifier.training = True
        return