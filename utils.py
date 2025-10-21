import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.manifold import TSNE

def save_features(process_features, save_name, save_dir):
    savepath = os.path.join(save_dir,save_name+'.pth')

    torch.save(process_features.cpu(), savepath)


def tsne_plot_save_dir(features, labels, result_dir, savename='tsne_visualization.png'):
    # 创建 t-SNE 模型并进行降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 获取唯一的标签类别
    unique_labels = np.unique(labels)

    # 定义颜色映射
    colors = ['#EAB38A', '#94AAD8']

    # 绘制可视化图形
    plt.figure(figsize=(8, 8))
    for label in unique_labels:
        # 获取属于当前类别的特征和对应的 t-SNE 降维结果
        label_features = features_tsne[labels == label]

        # 获取当前类别的颜色
        color = colors[label]

        # 绘制当前类别的散点图
        plt.scatter(label_features[:, 0], label_features[:, 1], color=color, label=f'Label {label}', alpha=0.8)


    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                    labelleft=False)

    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, savename)
    plt.savefig(save_path)


def create_task_flags_sperate(task, exif_initW=1, face_initW=0.1):
    semantic_tasks_2lvls = {
                            'iso': exif_initW, 'av': exif_initW, 'et': exif_initW, 'fl': exif_initW,
                            'makes': exif_initW, 'mm': exif_initW, 'em': exif_initW, 'wb': exif_initW, 'ep': exif_initW,
                            'face_coarse': face_initW, 'face_grained': face_initW,
                            }
    tasks = {}
    if task != 'all':
        tasks[task] = semantic_tasks_2lvls[task]
    else:
        tasks = semantic_tasks_2lvls
    return tasks

def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id.title(), weight[i])
    return weight_str


def predict_batch(tensor_batch):
    _, predicted_indices = torch.max(tensor_batch, dim=1)
    return predicted_indices

def accuracy_batch(predicted_indices, true_labels):
    correct = (predicted_indices == true_labels)
    acc = correct.sum().item() / true_labels.size(0)
    return acc