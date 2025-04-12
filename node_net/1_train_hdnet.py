import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import LambdaLR

sys.path.append(r"C:\Users\PC\PycharmProjects\thu_xwh\Codes")
from node_toolkit.hdnet import HDNet  # 使用修改后的 HDNet
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
    save_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Codes/Model/HDNet0411"
    os.makedirs(save_dir, exist_ok=True)

    # 超参数
    num_dimensions = 3
    batch_size = 2
    num_epochs = 200
    learning_rate = 1e-2
    k_folds = 5
    validation_interval = 1
    patience = 200

    # 定义新 HDNet 配置，适配 DFilter 的参数
    node_configs = {
        0: (4, 64, 64, 64),  # 输入节点：4通道图像
        1: (1, 64, 64, 64),  # 输入节点：1通道图像
        2: (32, 64, 64, 64),
        3: (32, 64, 64, 64),
        4: (64, 16, 16, 16),
        5: (128, 4, 4, 4),
        6: (128, 4, 4, 4),
        7: (128, 4, 4, 4),
        8: (128, 4, 4, 4),
        9: (4, 1, 1, 1),   # 输出节点：分类任务
        10: (4, 1, 1, 1),  # 输出节点：分类任务
        11: (1, 1, 1, 1),  # 输出节点：回归任务
        12: (2, 1, 1, 1),  # 输出节点：分类任务
        13: (2, 1, 1, 1),  # 输出节点：分类任务
        14: (2, 1, 1, 1),  # 输出节点：分类任务
        15: (3, 1, 1, 1),  # 输出节点：分类任务
        16: (2, 1, 1, 1),  # 输出节点：分类任务
    }
    hyperedge_configs = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [2],
            "params": {
                "convs": [(32, 3, 3, 3)],  # 输出32通道，3x3x3卷积核
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
                "convs": [(512, 3, 3, 3), (128, 3, 3, 3)],  # 多层卷积
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
                "feature_size": (4, 4, 4),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e5": {
            "src_nodes": [5],
            "dst_nodes": [9],
            "params": {
                "convs": [(32, 1, 1, 1)],  # 分类任务
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e6": {
            "src_nodes": [6],
            "dst_nodes": [10],
            "params": {
                "convs": [(32, 1, 1, 1)],  # 分类任务
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e7": {
            "src_nodes": [7],
            "dst_nodes": [11],
            "params": {
                "convs": [(32, 1, 1, 1)],  # 回归任务
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e8": {
            "src_nodes": [8],
            "dst_nodes": [12],
            "params": {
                "convs": [(32, 1, 1, 1)],  # 分类任务
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e9": {
            "src_nodes": [8],
            "dst_nodes": [13],
            "params": {
                "convs": [(32, 1, 1, 1)],  # 分类任务
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e10": {
            "src_nodes": [8],
            "dst_nodes": [14],
            "params": {
                "convs": [(32, 1, 1, 1)],  # 分类任务
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e11": {
            "src_nodes": [8],
            "dst_nodes": [15],
            "params": {
                "convs": [(32, 1, 1, 1)],  # 分类任务
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
        "e12": {
            "src_nodes": [8],
            "dst_nodes": [16],
            "params": {
                "convs": [(32, 1, 1, 1)],  # 分类任务
                "feature_size": (1, 1, 1),
                "in_p": "linear",
                "out_p": "linear",
            },
        },
    }
    in_nodes = [0, 1]
    out_nodes = [9, 10, 11, 12, 13, 14, 15, 16]

    # 外部实例化变换对象
    random_rotate = RandomRotate3D()
    random_flip = RandomFlip3D()
    random_shift = RandomShift3D()
    random_zoom = RandomZoom3D()
    min_max_normalize = MinMaxNormalize3D()
    z_score_normalize = ZScoreNormalize3D()

    # 每个节点的变换列表
    node_transforms = {
        0: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        1: [random_rotate, random_flip, random_shift, random_zoom],
    }

    # 任务与节点映射
    task_node_map = {
        "Feature_1": 9,   # 分类任务
        "Feature_2": 10,  # 分类任务
        "Feature_3": 11,  # 回归任务
        "Feature_4": 12,  # 分类任务
        "Feature_5": 13,  # 分类任务
        "Feature_6": 14,  # 分类任务
        "Feature_7": 15,  # 分类任务
        "Feature_8": 16,  # 分类任务
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
        0: ["0000", "0001", "0002", "0003"],  # 节点 0 的所有图像后缀
        1: ["0004"],                          # 节点 1 的所有图像后缀
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
        model = HDNet(
            node_configs,
            hyperedge_configs,
            in_nodes,
            out_nodes,
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
                        "node_configs": {str(k): list(v) for k, v in node_configs.items()},
                        "hyperedge_configs": {
                            k: {
                                "src_nodes": v["src_nodes"],
                                "dst_nodes": v["dst_nodes"],
                                "params": v["params"],
                            }
                            for k, v in hyperedge_configs.items()
                        },
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
