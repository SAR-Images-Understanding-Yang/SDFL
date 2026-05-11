import os
import re
import numpy as np
from typing import List, Dict, Tuple

def parse_analysis_file(file_path: str) -> Dict[str, float]:
    """
    解析单个analysis.txt文件，提取关键指标（含A-distance）
    :param file_path: analysis.txt文件路径
    :return: 包含指标的字典
    """
    metrics = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 1. 提取测试集最终accuracy（两个测试阶段，取最后一个的最终值）
            acc_pattern = r'acc=tensor\((\d+\.?\d*)\)'
            acc_matches = re.findall(acc_pattern, content)
            if acc_matches:
                # 识别测试阶段，取最后一个测试阶段的最终acc
                test_phases = []
                current_phase = []
                for acc_str in acc_matches:
                    current_phase.append(float(acc_str))
                    if float(acc_str) == 100.0 or len(current_phase) >= 23:
                        test_phases.append(current_phase)
                        current_phase = []
                if current_phase:
                    test_phases.append(current_phase)
                
                # 取最后一个测试阶段的最终acc（目标域测试结果）
                if test_phases:
                    metrics['accuracy'] = test_phases[-1][-1]
            
            # 2. 提取Macro Precision
            precision_pattern = r'Macro Precision: (\d+\.?\d*)'
            precision_match = re.search(precision_pattern, content)
            if precision_match:
                metrics['precision'] = float(precision_match.group(1))
            
            # 3. 提取Macro Recall
            recall_pattern = r'Macro Recall: (\d+\.?\d*)'
            recall_match = re.search(recall_pattern, content)
            if recall_match:
                metrics['recall'] = float(recall_match.group(1))
            
            # 4. 提取Macro F1 Score
            f1_pattern = r'Macro F1 Score: (\d+\.?\d*)'
            f1_match = re.search(f1_pattern, content)
            if f1_match:
                metrics['f1'] = float(f1_match.group(1))
            
            # 5. 提取最终A-distance（取全局最终值，而非每个epoch的A-dist）
            adist_pattern = r'A-distance = tensor\((\d+\.?\d*), device'
            adist_match = re.search(adist_pattern, content)
            if adist_match:
                metrics['a_distance'] = float(adist_match.group(1))
            else:
                # 备用匹配：如果没找到最终A-distance，尝试匹配epoch的A-dist（取最后一个epoch的）
                epoch_adist_pattern = r'epoch \d+ accuracy: \d+\.?\d* A-dist: (\d+\.?\d*)'
                epoch_adist_matches = re.findall(epoch_adist_pattern, content)
                if epoch_adist_matches:
                    metrics['a_distance'] = float(epoch_adist_matches[-1])
                
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {str(e)}")
    
    return metrics

def collect_all_experiments(root_dir: str) -> List[Dict[str, float]]:
    """
    遍历根文件夹下的所有子文件夹，收集所有实验的指标
    :param root_dir: 根文件夹路径
    :return: 所有实验的指标列表
    """
    all_metrics = []
    
    # 遍历根文件夹下的所有子文件夹
    for experiment_dir in os.listdir(root_dir):
        experiment_path = os.path.join(root_dir, experiment_dir)
        
        # 只处理文件夹
        if os.path.isdir(experiment_path):
            # 查找analysis.txt文件（支持文件名变体）
            for file_name in os.listdir(experiment_path):
                if file_name.lower().startswith('analysis') and file_name.endswith('.txt'):
                    analysis_file = os.path.join(experiment_path, file_name)
                    print(f"正在解析: {analysis_file}")
                    
                    metrics = parse_analysis_file(analysis_file)
                    if metrics:  # 只添加有效数据
                        all_metrics.append(metrics)
                        print(f"  提取的指标: {metrics}")
                    else:
                        print(f"  未提取到有效指标")
                    break  # 每个实验文件夹只处理一个analysis文件
    
    return all_metrics

def calculate_statistics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    计算所有指标的均值、方差、标准差
    :param metrics_list: 所有实验的指标列表
    :return: 统计结果
    """
    if not metrics_list:
        return {}
    
    # 获取所有可用的指标名称
    metric_names = metrics_list[0].keys()
    statistics = {}
    
    for metric in metric_names:
        # 收集该指标的所有数据
        values = [metrics[metric] for metrics in metrics_list if metric in metrics]
        if len(values) < 2:
            print(f"警告: 指标 {metric} 只有 {len(values)} 个有效数据点，无法计算统计量")
            continue
        
        # 计算统计量（样本方差/标准差，ddof=1）
        mean_val = np.mean(values)
        var_val = np.var(values, ddof=1)
        std_val = np.std(values, ddof=1)
        
        statistics[metric] = {
            'count': len(values),
            'mean': mean_val,
            'variance': var_val,
            'std': std_val
        }
    
    return statistics

def print_statistics_report(statistics: Dict[str, Dict[str, float]]):
    """
    打印统计报告
    """
    if not statistics:
        print("没有足够的有效数据生成统计报告")
        return
    
    print("\n" + "="*70)
    print("实验结果统计报告（含A-distance）")
    print("="*70)
    print(f"{'指标':<15} {'样本数':<8} {'均值':<12} {'方差':<12} {'标准差':<12}")
    print("-"*70)
    
    for metric, stats in statistics.items():
        # 美化指标名称显示
        metric_name = metric.replace('_', ' ').title() if '_' in metric else metric.title()
        print(f"{metric_name:<15} {stats['count']:<8} {stats['mean']:<12.4f} "
              f"{stats['variance']:<12.4f} {stats['std']:<12.4f}")
    
    print("="*70)

def save_statistics_to_csv(statistics: Dict[str, Dict[str, float]], output_file: str):
    """
    将统计结果保存到CSV文件
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['指标', '样本数', '均值', '方差', '标准差'])
        # 写入数据
        for metric, stats in statistics.items():
            # 美化指标名称
            metric_name = metric.replace('_', ' ').title() if '_' in metric else metric.title()
            writer.writerow([
                metric_name,
                stats['count'],
                round(stats['mean'], 4),
                round(stats['variance'], 4),
                round(stats['std'], 4)
            ])
    
    print(f"\n统计结果已保存到: {output_file}")

def main():
    # -------------------------- 配置参数 --------------------------
    ROOT_DIRECTORY = r"/home/xyang/code/SARCDG_results/SimulatedSARShip/ablation/6"  # 替换为你的实验根文件夹路径
    OUTPUT_CSV = r"/home/xyang/code/SARCDG_results/SimulatedSARShip/ablation/6/experiment_statistics_with_adistance.csv"  # 输出CSV文件名
    # --------------------------------------------------------------
    
    print(f"开始处理实验结果（含A-distance提取），根文件夹: {ROOT_DIRECTORY}")
    print("-"*70)
    
    # 1. 收集所有实验指标
    all_metrics = collect_all_experiments(ROOT_DIRECTORY)
    
    if not all_metrics:
        print("未找到任何有效实验数据")
        return
    
    print(f"\n共收集到 {len(all_metrics)} 个有效实验的指标")
    
    # 2. 计算统计量
    statistics = calculate_statistics(all_metrics)
    
    # 3. 打印报告
    print_statistics_report(statistics)
    
    # 4. 保存到CSV
    if statistics:
        save_statistics_to_csv(statistics, OUTPUT_CSV)
    
    # 附加说明
    # print("\n" + "="*70)
    # print("关于统计指标的选择:")
    # print("="*70)
    # print("1. 必须报告：均值（反映平均性能）")
    # print("2. 推荐报告：标准差（直观反映波动范围）")
    # print("3. A-distance说明：域自适应任务中核心指标，值越小域适配效果越好")
    # print("4. 实验次数建议：≥3次，统计结果更可靠")
    # print("5. 标准报告格式：指标名称: 均值 ± 标准差（例如：A-distance: 1.97 ± 0.02）")

if __name__ == "__main__":
    main()