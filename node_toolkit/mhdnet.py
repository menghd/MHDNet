import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import sys
sys.path.append(r"C:\Users\PC\PycharmProjects\thu_xwh\Codes")
from node_toolkit.hdnet import HDNet  

class MHDNet(nn.Module):
    """多子超图网络（MHDNet），将子 HDNet 作为模块处理全局节点输入和输出。

    Args:
        sub_networks: 子 HDNet 网络，键为网络名称，值为 HDNet 实例。
        node_mapping: 节点映射，格式为 [(global_node_id, sub_net_name, sub_node_id), ...]。
        in_nodes: 全局输入节点 ID 列表。
        out_nodes: 全局输出节点 ID 列表。
        num_dimensions: 维度（1D、2D 或 3D）。
    """
    def __init__(
        self,
        sub_networks: Dict[str, nn.Module],
        node_mapping: List[Tuple[int, str, int]],
        in_nodes: List[int],
        out_nodes: List[int],
        num_dimensions: int = 2,
    ):
        super().__init__()
        self.sub_networks = nn.ModuleDict(sub_networks)
        self.node_mapping = node_mapping
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes if isinstance(out_nodes, list) else [out_nodes]
        self.num_dimensions = num_dimensions

        self._validate_mapping()
        self._check_node_shapes()

    def _validate_mapping(self):
        """验证节点映射的有效性"""
        # 检查 node_mapping 的格式
        for mapping in self.node_mapping:
            if not isinstance(mapping, tuple) or len(mapping) != 3:
                raise ValueError(f"node_mapping 条目格式错误: {mapping}")
            global_node, sub_net_name, sub_node_id = mapping
            if sub_net_name not in self.sub_networks:
                raise ValueError(f"子网络 {sub_net_name} 未在 sub_networks 中定义")
            sub_net = self.sub_networks[sub_net_name]
            if sub_node_id not in sub_net.node_configs:
                raise ValueError(f"子网络 {sub_net_name} 中不存在节点 {sub_node_id}")

        # 检查全局输入节点
        mapped_global_nodes = {mapping[0] for mapping in self.node_mapping}
        for node in self.in_nodes:
            if node not in mapped_global_nodes:
                raise ValueError(f"全局输入节点 {node} 未在 node_mapping 中定义")
            # 确保全局输入节点映射到子网络的输入节点
            found = False
            for g_node, s_name, s_node_id in self.node_mapping:
                if g_node == node:
                    sub_net = self.sub_networks[s_name]
                    if s_node_id in sub_net.in_nodes:
                        found = True
                        break
            if not found:
                raise ValueError(f"全局输入节点 {node} 未映射到任何子网络的输入节点")

        # 检查全局输出节点
        for node in self.out_nodes:
            if node not in mapped_global_nodes:
                raise ValueError(f"全局输出节点 {node} 未在 node_mapping 中定义")
            # 确保全局输出节点映射到子网络的输出节点
            found = False
            for g_node, s_name, s_node_id in self.node_mapping:
                if g_node == node:
                    sub_net = self.sub_networks[s_name]
                    if s_node_id in sub_net.out_nodes:
                        found = True
                        break
            if not found:
                raise ValueError(f"全局输出节点 {node} 未映射到任何子网络的输出节点")

    def _check_node_shapes(self):
        """检查全局节点与子网络节点的形状有效性"""
        for global_node, sub_net_name, sub_node_id in self.node_mapping:
            sub_net = self.sub_networks[sub_net_name]
            shape = sub_net.node_configs[sub_node_id]
            if not shape or not all(isinstance(s, int) and s > 0 for s in shape):
                raise ValueError(f"节点 {global_node} 的映射节点 {sub_node_id} 形状无效: {shape}")

        # 检查同一全局节点映射的子节点形状一致
        shape_map = {}
        for global_node, sub_net_name, sub_node_id in self.node_mapping:
            sub_net = self.sub_networks[sub_net_name]
            shape = sub_net.node_configs[sub_node_id]
            if global_node in shape_map:
                if shape_map[global_node] != shape:
                    raise ValueError(
                        f"全局节点 {global_node} 映射到不同形状: "
                        f"{shape_map[global_node]} vs {shape}"
                    )
            else:
                shape_map[global_node] = shape

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播，通过子网络处理全局节点输入并收集输出。

        Args:
            inputs: 输入张量列表，对应 in_nodes。

        Returns:
            输出张量列表，对应 out_nodes。
        """
        if len(inputs) != len(self.in_nodes):
            raise ValueError(
                f"预期输入数量为 {len(self.in_nodes)}，实际得到 {len(inputs)}"
            )

        # 初始化全局节点特征
        global_features = {}
        for global_node, input_tensor in zip(self.in_nodes, inputs):
            global_features[global_node] = input_tensor

        # 初始化子网络输入
        sub_net_inputs = {
            name: [None] * len(net.in_nodes)
            for name, net in self.sub_networks.items()
        }

        # 填充全局输入到子网络
        for global_node in self.in_nodes:
            for g_node, sub_net_name, sub_node_id in self.node_mapping:
                if g_node == global_node:
                    sub_net = self.sub_networks[sub_net_name]
                    idx = sub_net.in_nodes.index(sub_node_id)
                    sub_net_inputs[sub_net_name][idx] = global_features[global_node]

        # 动态处理子网络依赖
        processed_nets = set()
        sub_net_outputs = {}
        while len(processed_nets) < len(self.sub_networks):
            progress = False
            for name, net in self.sub_networks.items():
                if name in processed_nets:
                    continue

                # 检查该子网络的所有输入是否都已准备好
                inputs_ready = True
                for sub_node_id in net.in_nodes:
                    global_node = None
                    for g_node, s_name, s_node_id in self.node_mapping:
                        if s_name == name and s_node_id == sub_node_id:
                            global_node = g_node
                            break
                    if global_node is None:
                        raise ValueError(f"子网络 {name} 的输入节点 {sub_node_id} 未在 node_mapping 中定义")

                    if global_node not in global_features:
                        src_net_name, src_node_id = None, None
                        for g_node, s_name, s_node_id in self.node_mapping:
                            if g_node == global_node and s_name != name:
                                src_net_name, src_node_id = s_name, s_node_id
                                break
                        if src_net_name is None or src_node_id is None:
                            raise ValueError(f"全局节点 {global_node} 未找到输出来源")
                        if src_net_name not in sub_net_outputs or src_node_id not in sub_net_outputs[src_net_name]:
                            inputs_ready = False
                            break

                if inputs_ready:
                    # 填充依赖的输入
                    for sub_node_id in net.in_nodes:
                        global_node = None
                        for g_node, s_name, s_node_id in self.node_mapping:
                            if s_name == name and s_node_id == sub_node_id:
                                global_node = g_node
                                break
                        idx = net.in_nodes.index(sub_node_id)
                        if global_node in global_features:
                            sub_net_inputs[name][idx] = global_features[global_node]
                        else:
                            src_net_name, src_node_id = None, None
                            for g_node, s_name, s_node_id in self.node_mapping:
                                if g_node == global_node and s_name != name:
                                    src_net_name, src_node_id = s_name, s_node_id
                                    break
                            sub_net_inputs[name][idx] = sub_net_outputs[src_net_name][src_node_id]

                    # 运行子网络
                    outputs = net(sub_net_inputs[name])
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    sub_net_outputs[name] = dict(zip(net.out_nodes, outputs))

                    # 更新全局节点特征
                    for sub_node_id, output_tensor in zip(net.out_nodes, outputs):
                        global_node = None
                        for g_node, s_name, s_node_id in self.node_mapping:
                            if s_name == name and s_node_id == sub_node_id:
                                global_node = g_node
                                break
                        if global_node is not None:
                            global_features[global_node] = output_tensor

                    processed_nets.add(name)
                    progress = True

            if not progress:
                raise ValueError("检测到循环依赖或无法满足的子网络输入")

        # 收集全局输出
        outputs = []
        for global_node in self.out_nodes:
            if global_node not in global_features:
                raise ValueError(f"全局输出节点 {global_node} 无输出")
            outputs.append(global_features[global_node])

        return outputs

def run_example(
    sub_network_configs: List[Tuple[Dict[int, Tuple[int, ...]], Dict[str, Dict], List[int], List[int]]],
    node_mapping: List[Tuple[int, str, int]],
    in_nodes: List[int],
    out_nodes: List[int],
    num_dimensions: int,
    input_shapes: List[Tuple[int, ...]],
    onnx_filename: str,
):
    """运行并导出 MHDNet 模型为 ONNX。

    Args:
        sub_network_configs: 子网络配置列表，每个元素为 (node_configs, hyperedge_configs, in_nodes, out_nodes)。
        node_mapping: 全局节点到子网络节点的映射，格式为 [(global_node_id, sub_net_name, sub_node_id), ...]。
        in_nodes: 全局输入节点 ID。
        out_nodes: 全局输出节点 ID。
        num_dimensions: 维度。
        input_shapes: 输入张量形状。
        onnx_filename: 导出的 ONNX 文件名。
    """
    sub_networks = {}
    for i, (node_configs, hyperedge_configs, sub_in_nodes, sub_out_nodes) in enumerate(sub_network_configs):
        sub_net = HDNet(node_configs, hyperedge_configs, sub_in_nodes, sub_out_nodes, num_dimensions)
        sub_networks[f"net{i+1}"] = sub_net

    model = MHDNet(
        sub_networks=sub_networks,
        node_mapping=node_mapping,
        in_nodes=in_nodes,
        out_nodes=out_nodes,
        num_dimensions=num_dimensions,
    )
    model.eval()

    inputs = [torch.randn(*shape) for shape in input_shapes]
    outputs = model(inputs)
    for node, out in zip(out_nodes, outputs):
        print(f"全局节点 {node} 的输出: {out.shape}")

    dynamic_axes = {
        **{f"input_{node}": {0: "batch_size"} for node in in_nodes},
        **{f"output_{node}": {0: "batch_size"} for node in out_nodes},
    }
    torch.onnx.export(
        model,
        inputs,
        onnx_filename,
        input_names=[f"input_{node}" for node in in_nodes],
        output_names=[f"output_{node}" for node in out_nodes],
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )
    print(f"模型已导出为 {onnx_filename}")

def example_mhdnet():
    # 子网络 1 配置
    node_configs1 = {
        0: (4, 64, 64, 64),
        1: (1, 64, 64, 64),
        2: (32, 8, 8, 8),
        3: (32, 8, 8, 8),
        4: (32, 8, 8, 8),
        5: (64, 1, 1, 1),
    }
    hyperedge_configs1 = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [2],
            "params": {
                "feature_size": (64, 64, 64),
                "out_p": 2,
                "convs": [(32, 3, 3, 3), (32, 3, 3, 3)],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
            },
        },
        "e2": {
            "src_nodes": [1],
            "dst_nodes": [3],
            "params": {
                "feature_size": (64, 64, 64),
                "out_p": 2,
                "convs": [(32, 3, 3, 3), (32, 3, 3, 3)],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
            },
        },
        "e3": {
            "src_nodes": [0, 1],
            "dst_nodes": [4],
            "params": {
                "feature_size": (64, 64, 64),
                "out_p": 2,
                "convs": [(32, 3, 3, 3), (32, 3, 3, 3)],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
            },
        },
        "e4": {
            "src_nodes": [4],
            "dst_nodes": [5],
            "params": {
                "feature_size": (8, 8, 8),
                "in_p": "linear",
                "out_p": "linear",
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
            },
        },
        "e5": {
            "src_nodes": [2, 3],
            "dst_nodes": [5],
            "params": {
                "feature_size": (8, 8, 8),
                "in_p": "linear",
                "out_p": "linear",
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
            },
        },
    }

    # 子网络 2 配置
    node_configs2 = {
        0: (64, 1, 1, 1),  # 输入
        1: (128, 1, 1, 1),  # 输出
        2: (128, 1, 1, 1),  # 输出
    }
    hyperedge_configs2 = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [1],
            "params": {
                "feature_size": (1, 1, 1),
                "convs": [],
            },
        },
        "e2": {
            "src_nodes": [0],
            "dst_nodes": [2],
            "params": {
                "feature_size": (1, 1, 1),
                "convs": [],
            },
        },
    }

    # 子网络 3 配置
    node_configs3 = node_configs2
    hyperedge_configs3 = hyperedge_configs2
    hyperedge_configs3.update({
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [1],
            "params": {
                "feature_size": (1, 1, 1),
                "convs": [(128,1,1,1)],
                "acts":["sigmoid"]
            },
        },
    })

    # 节点映射
    node_mapping = [
        (100, "net1", 0),  # 全局输入 → 子网络 1 节点 0
        (101, "net1", 1),  # 全局输入 → 子网络 1 节点 1
        (102, "net1", 5),  # 子网络 1 节点 5 输出
        (102, "net2", 0),  # 子网络 2 节点 0 输入
        (102, "net3", 0),  # 子网络 2 节点 0 输入
        (103, "net2", 1),  # 子网络 2 节点 1 → 全局输出
        (104, "net2", 2),  # 子网络 2 节点 2 → 全局输出
        (105, "net3", 1),  # 子网络 2 节点 2 → 全局输出
        (106, "net3", 2),  # 子网络 2 节点 2 → 全局输出
    ]

    # 运行示例
    run_example(
        sub_network_configs=[
            (node_configs1, hyperedge_configs1, [0, 1], [5]),  # 子网络 1
            (node_configs2, hyperedge_configs2, [0], [1, 2]),  # 子网络 2
            (node_configs3, hyperedge_configs3, [0], [1,2]),  # 子网络 2
        ],
        node_mapping=node_mapping,
        in_nodes=[100, 101],  # 全局输入
        out_nodes=[103, 104, 105, 106],  # 全局输出
        num_dimensions=3,
        input_shapes=[(2, 4, 64, 64, 64), (2, 1, 64, 64, 64)],  # 对应 100 和 101
        onnx_filename="MHDNet_example.onnx",
    )

if __name__ == "__main__":
    example_mhdnet()
