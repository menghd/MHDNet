import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import warnings

class DFilter(nn.Module):
    """动态滤波器，支持多层卷积、归一化和激活，保持空间形状不变。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        num_dimensions (int): 维度（1D、2D 或 3D）。
        convs (List[Optional[Tuple[int, ...]]]): 每层卷积配置，格式为 (out_channels, kernel_size...)，None 表示无卷积。
        norms (Optional[List[Optional[str]]]): 每层归一化类型，None 表示无归一化。
        acts (Optional[List[Optional[str]]]): 每层激活函数类型，None 表示无激活。
    """
    NORM_TYPES = {
        "instance": lambda dim, ch: getattr(nn, f"InstanceNorm{dim}d")(ch),
        "batch": lambda dim, ch: getattr(nn, f"BatchNorm{dim}d")(ch),
    }
    ACT_TYPES = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leakyrelu": lambda: nn.LeakyReLU(0.01, inplace=True),
        "sigmoid": lambda: nn.Sigmoid(),
        "softmax": lambda: nn.Softmax(dim=1),
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dimensions: int,
        convs: List[Optional[Tuple[int, ...]]],
        norms: Optional[List[Optional[str]]] = None,
        acts: Optional[List[Optional[str]]] = None,
    ):
        super().__init__()
        self.num_dimensions = num_dimensions
        conv_layer = getattr(nn, f"Conv{num_dimensions}d")
        layers = []

        # 默认值处理
        norms = norms if norms is not None else [None] * len(convs)
        acts = acts if acts is not None else [None] * len(convs)

        # 验证配置长度
        if not (len(convs) == len(norms) == len(acts)):
            raise ValueError(
                f"配置长度不一致：convs={len(convs)}, norms={len(norms)}, acts={len(acts)}"
            )

        current_channels = in_channels
        for i, (conv_config, norm_type, act_type) in enumerate(zip(convs, norms, acts)):
            if conv_config is not None:
                conv_out_channels, *kernel_size = conv_config
                padding = tuple(k // 2 for k in kernel_size)
                layers.append(
                    conv_layer(
                        current_channels,
                        conv_out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    )
                )
                current_channels = conv_out_channels

            if norm_type:
                norm_type = norm_type.lower()
                if norm_type not in self.NORM_TYPES:
                    raise ValueError(f"第 {i} 层不支持的归一化类型: {norm_type}")
                layers.append(self.NORM_TYPES[norm_type](self.num_dimensions, current_channels))

            if act_type:
                act_type = act_type.lower()
                if act_type not in self.ACT_TYPES:
                    raise ValueError(f"第 {i} 层不支持的激活函数: {act_type}")
                layers.append(self.ACT_TYPES[act_type]())

        # 确保输出通道匹配
        if current_channels != out_channels:
            layers.append(
                conv_layer(current_channels, out_channels, kernel_size=1, bias=False)
            )

        self.filter = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.filter(x)


class HDNet(nn.Module):
    """基于超边的网络，支持动态维度和灵活的节点连接。

    Args:
        node_configs (Dict[int, Tuple[int, ...]]): 节点配置，格式为 {node_id: (channels, size...)}。
        hyperedge_configs (Dict[str, Dict]): 超边配置，包含 src_nodes, dst_nodes 和 params。
        in_nodes (List[int]): 输入节点 ID 列表。
        out_nodes (List[int]): 输出节点 ID 列表。
        num_dimensions (int): 维度（1D、2D 或 3D）。
    """
    def __init__(
        self,
        node_configs: Dict[int, Tuple[int, ...]],
        hyperedge_configs: Dict[str, Dict],
        in_nodes: List[int],
        out_nodes: List[int],
        num_dimensions: int = 2,
    ):
        super().__init__()
        self.node_configs = node_configs
        self.hyperedge_configs = hyperedge_configs
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.num_dimensions = num_dimensions
        self.edges = nn.ModuleDict()
        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)

        self._validate_nodes()
        self._build_hyperedges()

    def _validate_nodes(self):
        """验证节点配置并移除未使用的节点"""
        all_nodes = set(self.node_configs.keys())
        used_nodes = set(self.in_nodes + self.out_nodes)
        for edge_config in self.hyperedge_configs.values():
            used_nodes.update(edge_config.get("src_nodes", []))
            used_nodes.update(edge_config.get("dst_nodes", []))
        unused_nodes = all_nodes - used_nodes
        if unused_nodes:
            warnings.warn(f"未使用的节点将被忽略: {unused_nodes}")
            self.node_configs = {k: v for k, v in self.node_configs.items() if k in used_nodes}

    def _compute_edge_channels(self, src_nodes: List[int], dst_nodes: List[int]) -> Tuple[int, int]:
        """计算超边的输入和输出通道数"""
        in_channels = sum(self.node_configs[src][0] for src in src_nodes)
        out_channels = sum(self.node_configs[dst][0] for dst in dst_nodes)
        return in_channels, out_channels

    def _build_hyperedges(self):
        """构建超边并初始化 DFilter"""
        for edge_id, edge_config in self.hyperedge_configs.items():
            src_nodes = edge_config.get("src_nodes", [])
            dst_nodes = edge_config.get("dst_nodes", [])
            params = edge_config.get("params", {})
            in_channels, out_channels = self._compute_edge_channels(src_nodes, dst_nodes)
            convs = params.get("convs", [(in_channels, 3, 3)])
            norms = params.get("norms")
            acts = params.get("acts")

            self.edges[edge_id] = DFilter(
                in_channels, out_channels, self.num_dimensions, convs, norms, acts
            )
            for src in src_nodes:
                self.out_edges[src].append(edge_id)
            for dst in dst_nodes:
                self.in_edges[dst].append(edge_id)

    def _get_interpolate_mode(self, p: str) -> str:
        """根据维度和插值类型返回 F.interpolate 的 mode 参数"""
        mode_map = {
            1: {"linear": "linear", "nearest": "nearest"},
            2: {"linear": "bilinear", "nearest": "nearest"},
            3: {"linear": "trilinear", "nearest": "nearest"},
        }
        return mode_map[self.num_dimensions][p]

    def _power_interpolate(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, ...]],
        p: Union[float, str],
    ) -> torch.Tensor:
        """使用 p 次幂插值调整张量大小，支持线性插值、最近邻、最大/平均池化。

        Args:
            x: 输入张量，形状为 (batch, channels, *spatial_dims)。
            target_size: 目标空间大小。
            p: 幂次参数，可以是浮点数（p 次幂插值）、'max'（最大池化）、'avg'（平均池化）、'nearest'（最近邻）、'linear'(线性)。

        Returns:
            调整大小后的张量。
        """
        if not target_size or x.shape[2:] == tuple(target_size):
            return x

        p = p.lower() if isinstance(p, str) else p
        if p in ("max", "avg"):
            pool_layer = getattr(nn, f"Adaptive{p.title()}Pool{self.num_dimensions}d")
            return pool_layer(target_size)(x)
        if p in ("nearest", "linear"):
            return F.interpolate(
                x,
                size=target_size,
                mode=self._get_interpolate_mode(p),
                align_corners=p == "linear",
            )

        # p 次幂插值
        spatial_dims = tuple(range(2, 2 + self.num_dimensions))
        min_vals = torch.amin(x, dim=spatial_dims, keepdim=True)
        x_nonneg = torch.clamp(x - min_vals, min=0.0)
        x_pow = torch.pow(x_nonneg, p)
        x_resized = F.interpolate(
            x_pow,
            size=target_size,
            mode=self._get_interpolate_mode("linear"),
            align_corners=True,
        )
        x_root = torch.pow(x_resized, 1.0 / p)
        min_vals_resized = F.interpolate(
            min_vals,
            size=target_size,
            mode=self._get_interpolate_mode("linear"),
            align_corners=True,
        )
        return x_root + min_vals_resized

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """前向传播。

        Args:
            inputs: 输入张量列表，对应 in_nodes。

        Returns:
            输出张量列表，对应 out_nodes。
        """
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("输入必须是张量列表或元组")
        if len(inputs) != len(self.in_nodes):
            raise ValueError(f"预期输入数量为 {len(self.in_nodes)}，实际得到 {len(inputs)}")

        # 初始化特征字典
        features = dict(zip(self.in_nodes, inputs))
        processed_nodes = set(self.in_nodes)
        all_nodes = set(self.node_configs.keys())

        # 逐步处理节点
        while processed_nodes != all_nodes:
            progress = False
            for node in all_nodes - processed_nodes:
                in_edge_ids = self.in_edges[node]
                if not in_edge_ids:
                    continue

                # 检查所有输入边是否就绪
                if not all(
                    all(
                        src in processed_nodes
                        for src in self.hyperedge_configs[edge_id].get("src_nodes", [])
                    )
                    for edge_id in in_edge_ids
                ):
                    continue

                # 收集节点输入
                node_inputs = []
                for edge_id in in_edge_ids:
                    edge_config = self.hyperedge_configs[edge_id]
                    src_nodes = edge_config.get("src_nodes", [])
                    dst_nodes = edge_config.get("dst_nodes", [])
                    params = edge_config.get("params", {})
                    feature_size = params.get("feature_size")
                    in_p = params.get("in_p", "linear")
                    out_p = params.get("out_p", "linear")

                    # 调整输入特征大小并拼接
                    src_features = [
                        self._power_interpolate(features[src], feature_size, in_p)
                        for src in src_nodes
                    ]
                    input_feat = torch.cat(src_features, dim=1)
                    output = self.edges[edge_id](input_feat)

                    # 分割输出并调整大小
                    channel_sizes = [self.node_configs[dst][0] for dst in dst_nodes]
                    split_outputs = torch.split(output, channel_sizes, dim=1)
                    dst_features = {
                        dst: self._power_interpolate(feat, self.node_configs[dst][1:], out_p)
                        for dst, feat in zip(dst_nodes, split_outputs)
                    }

                    if node in dst_features:
                        node_inputs.append(dst_features[node])

                if node_inputs:
                    features[node] = sum(node_inputs)  # 使用加法连接
                    processed_nodes.add(node)
                    progress = True
                else:
                    raise ValueError(f"节点 {node} 没有有效输入")

            if not progress:
                raise RuntimeError("图中存在无法解析的依赖，可能包含环或孤立节点")

        # 收集输出
        outputs = [features[node] for node in self.out_nodes]
        if len(outputs) != len(self.out_nodes):
            missing = [node for node in self.out_nodes if node not in features]
            raise ValueError(f"输出节点 {missing} 未被计算")
        return outputs
