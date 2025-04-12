import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy

sys.path.append(r"C:\Users\PC\PycharmProjects\thu_xwh\Codes")
from node_toolkit.mhdnet import MHDNet, HDNet
from node_toolkit.node_dataset import NodeDataset
from node_toolkit.node_loss import focal_loss, mse_loss
from node_toolkit.node_utils import train, validate
from node_toolkit.node_transform import (
    MinMaxNormalize3D,
    ZScoreNormalize3D,
    RandomRotate3D,
    RandomFlip3D,
    RandomShift3D,
    RandomZoom3D,
)

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 数据路径
    image_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Data/TrainNiigzCsvData/imagesTr"
    label_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Data/TrainNiigzCsvData/labelsTr"

    # 保存路径
    save_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Codes/Model/MHDNet0412"
    os.makedirs(save_dir, exist_ok=True)

    # 超参数
    num_dimensions = 3
    batch_size = 2
    num_epochs = 200
    learning_rate = 1e-2
    k_folds = 5
    validation_interval = 1
    patience = 200

    # 子网络1（主干，复用 HDNet 配置，去除最后任务分支）
    node_configs1 = {
        0: (4, 64, 64, 64),  # 输入节点：4通道图像
        1: (1, 64, 64, 64),  # 输入节点：1通道图像
        2: (32, 64, 64, 64),
        3: (32, 64, 64, 64),
        4: (64, 16, 16, 16),
        5: (128, 4, 4, 4),
        6: (128, 4, 4, 4),
        7: (128, 4, 4, 4),
        8: (128, 4, 4, 4),
    }
    hyperedge_configs1 = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [2],
            "params": {
                "convs": [(32, 3, 3, 3)],
                "norms": ["instance"],
                "acts": ["leakyrelu"],
                "feature_size": (64, 64, 64),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e2": {
            "src_nodes": [1],
            "dst_nodes": [3],
            "params": {
                "convs": [(32, 3, 3, 3)],
                "norms": ["instance"],
                "acts": ["leakyrelu"],
                "feature_size": (64, 64, 64),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e3": {
            "src_nodes": [2, 3],
            "dst_nodes": [4],
            "params": {
                "convs": [(64, 3, 3, 3)],
                "norms": ["instance"],
                "acts": ["leakyrelu"],
                "feature_size": (16, 16, 16),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e4": {
            "src_nodes": [4],
            "dst_nodes": [5, 6, 7, 8],
            "params": {
                "convs": [(512, 3, 3, 3), (128, 3, 3, 3)],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
                "feature_size": (4, 4, 4),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
    }
    in_nodes1 = [0, 1]
    out_nodes1 = [5, 6, 7, 8]

    # 子网络2（回归分支）
    node_configs2 = {
        0: (128, 4, 4, 4),  # 输入，连接子网络1的输出
        1: (1, 1, 1, 1),   # 输出：回归任务
    }
    hyperedge_configs2 = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [1],
            "params": {
                "convs": [(32, 1, 1, 1)],
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
    }
    in_nodes2 = [0]
    out_nodes2 = [1]

    # 子网络3（分类分支，支持多分类任务）
    node_configs3 = {
        0: (128, 4, 4, 4),  # 输入，连接子网络1的输出
        1: (4, 1, 1, 1),   # Feature_1：分类任务，4类
        2: (4, 1, 1, 1),   # Feature_2：分类任务，4类
        3: (2, 1, 1, 1),   # Feature_4：分类任务，2类
        4: (2, 1, 1, 1),   # Feature_5：分类任务，2类
        5: (2, 1, 1, 1),   # Feature_6：分类任务，2类
        6: (3, 1, 1, 1),   # Feature_7：分类任务，3类
        7: (2, 1, 1, 1),   # Feature_8：分类任务，2类
    }
    hyperedge_configs3 = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [1, 2, 3, 4, 5, 6, 7],
            "params": {
                "convs": [(32, 1, 1, 1)],
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
    }
    in_nodes3 = [0]
    out_nodes3 = [1, 2, 3, 4, 5, 6, 7]

    # 全局节点映射
    node_mapping = [
        (100, "net1", 0),  # 全局输入：4通道图像
        (101, "net1", 1),  # 全局输入：1通道图像
        (102, "net1", 5),  # 子网络1输出，连接到子网络2和3
        (102, "net2", 0),  # 子网络2输入
        (102, "net3", 0),  # 子网络3输入
        (103, "net2", 1),  # 全局输出：回归任务（Feature_3）
        (104, "net3", 1),  # 全局输出：分类任务（Feature_1）
        (105, "net3", 2),  # 全局输出：分类任务（Feature_2）
        (106, "net3", 3),  # 全局输出：分类任务（Feature_4）
        (107, "net3", 4),  # 全局输出：分类任务（Feature_5）
        (108, "net3", 5),  # 全局输出：分类任务（Feature_6）
        (109, "net3", 6),  # 全局输出：分类任务（Feature_7）
        (110, "net3", 7),  # 全局输出：分类任务（Feature_8）
    ]

    # 子网络配置字典
    sub_networks_configs = {
        "net1": (node_configs1, hyperedge_configs1, in_nodes1, out_nodes1),
        "net2": (node_configs2, hyperedge_configs2, in_nodes2, out_nodes2),
        "net3": (node_configs3, hyperedge_configs3, in_nodes3, out_nodes3),
    }

    # 子网络实例化
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes) in sub_networks_configs.items()
    }

    # 动态生成配置字典
    config_dict = {
        name: {
            "node_configs": {str(k): list(v) for k, v in node_configs.items()},
            "hyperedge_configs": deepcopy(hyperedge_configs),
            "in_nodes": in_nodes,
            "out_nodes": out_nodes,
        }
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes) in sub_networks_configs.items()
    }

    # 全局输入输出节点
    in_nodes = [100, 101]
    out_nodes = [103, 104, 105, 106, 107, 108, 109, 110]

    # 外部实例化变换对象
    random_rotate = RandomRotate3D()
    random_flip = RandomFlip3D()
    random_shift = RandomShift3D()
    random_zoom = RandomZoom3D()
    min_max_normalize = MinMaxNormalize3D()
    z_score_normalize = ZScoreNormalize3D()

    # 每个节点的变换列表
    node_transforms = {
        100: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        101: [random_rotate, random_flip, random_shift, random_zoom],
    }

    # 任务与节点映射
    task_node_map = {
        "Feature_1": 104,  # 分类任务，4类
        "Feature_2": 105,  # 分类任务，4类
        "Feature_3": 103,  # 回归任务
        "Feature_4": 106,  # 分类任务，2类
        "Feature_5": 107,  # 分类任务，2类
        "Feature_6": 108,  # 分类任务，2类
        "Feature_7": 109,  # 分类任务，3类
        "Feature_8": 110,  # 分类任务，2类
    }

    cls_features = ['Feature_1', 'Feature_2', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8']
    reg_features = ['Feature_3']
    lambda_weights = {'cls': [0.4, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05], 'reg': [0.1]}
    focal_params = {
        'Feature_1': ([1.0, 1.0, 1.0, 1.0], 2),
        'Feature_2': ([1.0, 1.0, 1.0, 1.0], 2),
        'Feature_4': ([1.0, 1.0], 2),
        'Feature_5': ([1.0, 1.0], 2),
        'Feature_6': ([1.0, 1.0], 2),
        'Feature_7': ([1.0, 1.0, 1.0], 2),
        'Feature_8': ([1.0, 1.0], 2),
    }

    # 输入节点图像后缀映射
    node_image_mappings = {
        100: ["0000", "0001", "0002", "0003"],  # 节点 100 的所有图像后缀
        101: ["0004"],                         # 节点 101 的所有图像后缀
    }

    # 创建所有输入节点的数据集
    datasets = {
        node: NodeDataset(
            image_dir,
            label_dir,
            node_image_mappings[node],
            node_transforms[node],
            cls_features,
            reg_features,
        )
        for node in in_nodes
    }

    # 获取所有节点的共同 case_ids
    all_case_ids = set.intersection(*(set(datasets[node].case_ids) for node in in_nodes))
    if not all_case_ids:
        raise ValueError("No common case_ids found across all input nodes!")
    all_case_ids = sorted(list(all_case_ids))

    # K折交叉验证
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_case_ids)):
        print(f"Fold {fold + 1}")

        # 获取训练和验证的 case_ids
        train_case_ids = [all_case_ids[idx] for idx in train_ids]
        val_case_ids = [all_case_ids[idx] for idx in val_ids]
        split_info = {
            "fold": fold + 1,
            "train_case_ids": train_case_ids,
            "val_case_ids": val_case_ids,
            "train_count": len(train_case_ids),
            "val_count": len(val_case_ids),
        }
        split_save_path = os.path.join(save_dir, f"fold_{fold + 1}_split.json")
        with open(split_save_path, "w") as f:
            json.dump(split_info, f, indent=4)
        print(f"Data split saved to {split_save_path}")

        # 为每个节点创建训练和验证的索引
        dataloaders_train = {}
        dataloaders_val = {}
        for node in in_nodes:
            node_case_ids = datasets[node].case_ids
            train_indices = [
                node_case_ids.index(case_id)
                for case_id in train_case_ids
                if case_id in node_case_ids
            ]
            val_indices = [
                node_case_ids.index(case_id)
                for case_id in val_case_ids
                if case_id in node_case_ids
            ]

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_indices)
            dataloaders_train[node] = DataLoader(
                datasets[node],
                batch_size=batch_size,
                sampler=train_subsampler,
                num_workers=16,
            )
            dataloaders_val[node] = DataLoader(
                datasets[node],
                batch_size=batch_size,
                sampler=val_subsampler,
                num_workers=16,
            )

        # 模型、优化器、调度器
        model = MHDNet(
            sub_networks=sub_networks,
            node_mapping=node_mapping,
            in_nodes=in_nodes,
            out_nodes=out_nodes,
            num_dimensions=num_dimensions,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)

        # 损失函数
        criterion_cls = (
            lambda output, target, feature: focal_loss(
                output,
                target,
                alpha=focal_params[feature][0],
                gamma=focal_params[feature][1],
            )
        )
        criterion_reg = mse_loss

        # 早停
        best_val_loss = float("inf")
        epochs_no_improve = 0
        log = {"fold": fold + 1, "epochs": []}

        # 训练循环
        for epoch in range(num_epochs):
            train_loss, train_cls_loss, train_reg_loss = train(
                model,
                dataloaders_train,
                optimizer,
                criterion_cls,
                criterion_reg,
                lambda_weights,
                cls_features,
                reg_features,
                task_node_map,
                out_nodes,
                epoch,
                num_epochs,
            )

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_cls_loss": train_cls_loss,
                "train_reg_loss": train_reg_loss,
            }

            if (epoch + 1) % validation_interval == 0:
                val_loss, val_cls_loss, val_reg_loss, metrics = validate(
                    model,
                    dataloaders_val,
                    criterion_cls,
                    criterion_reg,
                    lambda_weights,
                    cls_features,
                    reg_features,
                    task_node_map,
                    out_nodes,
                    epoch,
                    num_epochs,
                    focal_params
                )

                epoch_log.update(
                    {
                        "val_loss": val_loss,
                        "val_cls_loss": val_cls_loss,
                        "val_reg_loss": val_reg_loss,
                        "metrics": metrics,
                    }
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    save_path = os.path.join(save_dir, f"model_fold{fold + 1}_best.pth")
                    torch.save(model.state_dict(), save_path)
                    config = {
                        "sub_networks": config_dict,
                        "node_mapping": node_mapping,
                        "in_nodes": in_nodes,
                        "out_nodes": out_nodes,
                        "num_dimensions": num_dimensions,
                        "task_node_map": task_node_map,
                        "cls_features": cls_features,
                        "reg_features": reg_features,
                        "lambda_weights": lambda_weights,
                        "focal_params": focal_params,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "k_folds": k_folds,
                        "validation_interval": validation_interval,
                        "patience": patience,
                        "node_image_mappings": node_image_mappings,
                        "node_transforms": {
                            str(k): [t.__class__.__name__ for t in v]
                            for k, v in node_transforms.items()
                        },
                    }
                    config_save_path = os.path.join(
                        save_dir, f"model_config_fold{fold + 1}.json"
                    )
                    with open(config_save_path, "w") as f:
                        json.dump(config, f, indent=4)
                    print(f"Model saved to {save_path}, Config saved to {config_save_path}")
                else:
                    epochs_no_improve += validation_interval
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            scheduler.step()
            log["epochs"].append(epoch_log)

        log_save_path = os.path.join(save_dir, f"training_log_fold{fold + 1}.json")
        with open(log_save_path, "w") as f:
            json.dump(log, f, indent=4)
        print(f"Training log saved to {log_save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
