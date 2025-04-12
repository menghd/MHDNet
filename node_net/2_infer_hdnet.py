import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import sys
import json

sys.path.append(r"C:/Users/PC/PycharmProjects/thu_xwh/Codes/node_toolkit")
from Codes.node_toolkit.node_dataset import NodeDataset
from Codes.node_toolkit.hdnet import HDNet
from Codes.node_toolkit.node_transform import MinMaxNormalize3D

def inference(model, data_loader_dict, cls_features, reg_features, task_node_map, return_node, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        data_iterators = {node: iter(loader) for node, loader in data_loader_dict.items()}
        num_batches = min(len(loader) for loader in data_loader_dict.values())

        for batch_idx in range(num_batches):
            inputs_list = []
            batch_case_ids = None
            for node in sorted(data_iterators.keys()):
                images, _ = next(data_iterators[node])
                inputs_list.extend([img.to(device) for img in images])
                if batch_case_ids is None:
                    start_idx = batch_idx * data_loader_dict[node].batch_size
                    end_idx = min(start_idx + data_loader_dict[node].batch_size, len(data_loader_dict[node].dataset))
                    batch_case_ids = data_loader_dict[node].dataset.case_ids[start_idx:end_idx]

            outputs = model(inputs_list)

            cls_preds = {feature: [] for feature in cls_features}
            reg_preds = {feature: [] for feature in reg_features}

            for feature in cls_features:
                node_idx = return_node.index(task_node_map[feature])
                cls_output = outputs[node_idx]
                if cls_output.dim() > 2:
                    cls_output = cls_output.view(cls_output.size(0), cls_output.size(1))
                cls_pred = torch.argmax(cls_output, dim=1).cpu().numpy()
                cls_preds[feature] = cls_pred

            for feature in reg_features:
                node_idx = return_node.index(task_node_map[feature])
                reg_output = outputs[node_idx]
                if reg_output.dim() > 2:
                    reg_output = reg_output.view(reg_output.size(0), reg_output.size(1))
                reg_pred = reg_output.cpu().numpy().flatten()
                reg_preds[feature] = reg_pred

            for idx, case_id in enumerate(batch_case_ids):
                result_dict = {feature: [int(cls_preds[feature][idx])] for feature in cls_features}
                result_dict.update({feature: [float(reg_preds[feature][idx])] for feature in reg_features})
                result_df = pd.DataFrame(result_dict)
                output_path = os.path.join(output_dir, f'case_{case_id}.csv')
                result_df.to_csv(output_path, index=False)
                print(f"Saved predictions for case_{case_id} to {output_path}")

def load_model_and_config(save_dir, fold, device):
    config_path = os.path.join(save_dir, f'model_config_fold{fold}.json')
    weight_path = os.path.join(save_dir, f'model_fold{fold}_best.pth')

    with open(config_path, 'r') as f:
        config = json.load(f)

    node_configs = {int(k): tuple(v) for k, v in config["node_configs"].items()}
    hyperedge_dict = {
        edge_id: (value["src_nodes"], value["dst_nodes"], value["params"])
        for edge_id, value in config["hyperedge_dict"].items()
    }

    model = HDNet(node_configs, hyperedge_dict, config["in_nodes"], config["out_nodes"], 
                  num_dimensions=config["num_dimensions"]).to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()
    return model, config

def main():
    image_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Data/TrainNiigzCsvData/imagesTs"
    label_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Data/TrainNiigzCsvData/labelsTs"
    save_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Codes/Model/HDNet0401"
    fold = 1
    batch_size = 8
    os.makedirs(label_dir, exist_ok=True)

    model, config = load_model_and_config(save_dir, fold, device)
    print(f"Model and config loaded for fold {fold}")

    task_node_map = config["task_node_map"]
    cls_features = config["cls_features"]
    reg_features = config["reg_features"]
    return_node = config["out_nodes"]
    node_image_mappings = config["node_image_mappings"]
    in_nodes = config["in_nodes"]

    # 在外部实例化变换对象
    min_max_normalize = MinMaxNormalize3D()

    # 为每个节点指定变换列表
    node_transforms = {
        0: [min_max_normalize],  # 节点 0（四模态）仅归一化
        1: [min_max_normalize]   # 节点 1（分割图）仅归一化
    }

    # 创建数据集
    datasets = {
        node: NodeDataset(
            image_dir, label_dir, node_image_mappings[node], node_transforms[node], 
            cls_features, reg_features, inference_mode=True
        ) for node in in_nodes
    }
    data_loader_dict = {
        node: DataLoader(datasets[node], batch_size=batch_size, shuffle=False, num_workers=16) 
        for node in in_nodes
    }

    inference(model, data_loader_dict, cls_features, reg_features, task_node_map, return_node, label_dir)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
