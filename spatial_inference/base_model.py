from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torchvision.models import resnet101, resnet50, resnet34, resnet18

# from torch_sparse import SparseTensor, set_diag
from torch_geometric.typing import SparseTensor, torch_sparse

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

class GATv2Conv(MessagePassing):
    _alpha: OptTensor
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        n_history: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_history = n_history
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, n_history, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        assert isinstance(x, Tensor)
        assert x.dim() == 3

        x_l = self.lin_l(x).view(-1, self.n_history, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, self.n_history, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.n_history, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        # if edge_attr is not None:
        #     if edge_attr.dim() == 1:
        #         edge_attr = edge_attr.view(-1, 1)
        #     assert self.lin_edge is not None
        #     edge_attr = self.lin_edge(edge_attr)
        #     edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        #     x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

"""
Some basic operations used in resnet. Here we modify some of them.
"""

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample_style=0, residual=True):
        """
        downsample_style: [0, 1, 2], 0 means no downsample, 1 means, max pooling, 2 means conv 1x1 block downsample.
        """
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample_style = downsample_style
        self.downsample = None
        if downsample_style == 1:
            self.downsample = nn.MaxPool2d(2, 2)
        elif downsample_style == 2:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        self.residual = residual
    def forward(self, x):
        old_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample_style != 0 and self.residual:
            old_x = self.downsample(old_x)

        if self.residual:
            x += old_x
            x = self.relu(x)

        return x


class ImageEncoderNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageEncoderNet, self).__init__()
        self.network1 = torch.nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.network2 = torch.nn.Sequential(
            ResBlock(hidden_dim, hidden_dim),
            ResBlock(hidden_dim, output_dim, stride=2, downsample_style=2),
            ResBlock(output_dim, output_dim)
        )


    def forward(self, x):
        x = self.network1(x)
        return self.network2(x)


# TODO: original verison, can be modified.
class ImageQNet(nn.Module):
    def __init__(self, input_dim):
        super(ImageQNet, self).__init__()
        self.q_enc = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim // 4, kernel_size=1),
            nn.BatchNorm2d(input_dim // 4),
        )

    def forward(self, x):
        (batch_size, channels, width, height) = x.size()
        x_q = self.q_enc(x).view(batch_size, -1, width * height).permute(0,2,1)
        return x_q

class ImageKNet(nn.Module):
    def __init__(self, input_dim):
        super(ImageKNet, self).__init__()
        self.k_enc = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim // 4, kernel_size=1),
            nn.BatchNorm2d(input_dim // 4),
        )


    def forward(self, x):
        (batch_size, channels, width, height) = x.size()
        x_k = self.k_enc(x).view(batch_size, -1, width * height)
        return x_k

# TODO: original verison, can be modified.
class ImageVNet(nn.Module):
    def __init__(self, input_dim):
        super(ImageVNet, self).__init__()
        self.v_enc = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
        )

    def forward(self, x):
        (batch_size, channels, width, height) = x.size()
        return self.v_enc(x).view(batch_size, -1, width * height)


# TODO: original verison, can be modified.
class ImageCrossAttentionNet(nn.Module):
    def __init__(self, input_dim):
        super(ImageCrossAttentionNet, self).__init__()
        self.enc_poi = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        )
        self.enc_point = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        )
        self.enc_pop = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        )
        self.input_dim = input_dim

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, poi_q, poi_k, poi_v, point_q, point_k, point_v, pop_q, pop_k, pop_v, x_poi, x_point, x_pop):
        (batch_size, channels, width, height) = x_poi.size()

        poi_poi = self.softmax(torch.bmm(poi_q, poi_k)).permute(0, 2, 1)
        poi_point = self.softmax(torch.bmm(poi_q, point_k)).permute(0, 2, 1)
        poi_pop = self.softmax(torch.bmm(poi_q, pop_k)).permute(0, 2, 1)

        point_poi = self.softmax(torch.bmm(point_q, poi_k)).permute(0, 2, 1)
        point_point = self.softmax(torch.bmm(point_q, point_k)).permute(0, 2, 1)
        point_pop = self.softmax(torch.bmm(point_q, pop_k)).permute(0, 2, 1)

        pop_poi = self.softmax(torch.bmm(pop_q, poi_k)).permute(0, 2, 1)
        pop_point = self.softmax(torch.bmm(pop_q, point_k)).permute(0, 2, 1)
        pop_pop = self.softmax(torch.bmm(pop_q, pop_k)).permute(0, 2, 1)

        poi_poi = self.enc_poi(torch.bmm(poi_v, poi_poi).view(batch_size, channels, width, height)) + x_poi
        poi_point = self.enc_poi(torch.bmm(poi_v, poi_point).view(batch_size, channels, width, height)) + x_poi
        poi_pop = self.enc_poi(torch.bmm(poi_v, poi_pop).view(batch_size, channels, width, height)) + x_poi

        point_poi = self.enc_point(torch.bmm(point_v, point_poi).view(batch_size, channels, width, height)) + x_point
        point_point = self.enc_point(torch.bmm(point_v, point_point).view(batch_size, channels, width, height)) + x_point
        point_pop = self.enc_point(torch.bmm(point_v, point_pop).view(batch_size, channels, width, height)) + x_point

        pop_poi = self.enc_pop(torch.bmm(pop_v, pop_poi).view(batch_size, channels, width, height)) + x_pop
        pop_point = self.enc_pop(torch.bmm(pop_v, pop_point).view(batch_size, channels, width, height)) + x_pop
        pop_pop = self.enc_pop(torch.bmm(pop_v, pop_pop).view(batch_size, channels, width, height)) + x_pop

        return torch.concat([poi_poi, poi_point, poi_pop], dim=1), \
               torch.concat([point_poi, point_point, point_pop], dim=1), \
               torch.concat([pop_poi, pop_point, pop_pop], dim=1)


class ImageConnectorAtt2Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageConnectorAtt2Dense, self).__init__()
        self.enc_poi = nn.Sequential(
            ResBlock(input_dim, output_dim, stride=2, downsample_style=2),
            ResBlock(output_dim, output_dim),
        )
        self.enc_point = nn.Sequential(
            ResBlock(input_dim, output_dim, stride=2, downsample_style=2),
            ResBlock(output_dim, output_dim),
        )
        self.enc_pop = nn.Sequential(
            ResBlock(input_dim, output_dim, stride=2, downsample_style=2),
            ResBlock(output_dim, output_dim),
        )

    def forward(self, x_poi, x_point, x_pop):
        return self.enc_poi(x_poi), self.enc_point(x_point), self.enc_pop(x_pop)


class ImageDenseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageDenseNet, self).__init__()
        self.poi_network1 = ResBlock(input_dim * 3, hidden_dim, stride=2, downsample_style=2, residual=False)
        self.point_network1 = ResBlock(input_dim * 3, hidden_dim, stride=2, downsample_style=2, residual=False)
        self.pop_network1 = ResBlock(input_dim * 3, hidden_dim, stride=2, downsample_style=2, residual=False)

        self.poi_network2 = ResBlock(hidden_dim * 3, hidden_dim, residual=False)
        self.point_network2 = ResBlock(hidden_dim * 3, hidden_dim, residual=False)
        self.pop_network2 = ResBlock(hidden_dim * 3, hidden_dim, residual=False)

        self.poi_network3 = ResBlock(hidden_dim * 3, output_dim, stride=2, downsample_style=2, residual=False)
        self.point_network3 = ResBlock(hidden_dim * 3, output_dim, stride=2, downsample_style=2, residual=False)
        self.pop_network3 = ResBlock(hidden_dim * 3, output_dim, stride=2, downsample_style=2, residual=False)

        self.downsample1 = nn.Sequential(
            conv1x1(input_dim, hidden_dim, stride=2),
            nn.BatchNorm2d(hidden_dim)
        )
        self.downsample2 = nn.Sequential(
            conv1x1(hidden_dim, output_dim, stride=2),
            nn.BatchNorm2d(output_dim)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_poi, x_point, x_pop):
        x_poi_new = torch.concat([x_poi, x_point, x_pop], dim=1)
        x_point_new = torch.concat([x_point, x_pop, x_poi], dim=1)
        x_pop_new = torch.concat([x_pop, x_poi, x_point], dim=1)
        x_poi = self.relu(self.poi_network1(x_poi_new) + self.downsample1(x_poi))
        x_point = self.relu(self.point_network1(x_point_new) + self.downsample1(x_point))
        x_pop = self.relu(self.pop_network1(x_pop_new) + self.downsample1(x_pop))

        x_poi_new = torch.concat([x_poi, x_point, x_pop], dim=1)
        x_point_new = torch.concat([x_point, x_pop, x_poi], dim=1)
        x_pop_new = torch.concat([x_pop, x_poi, x_point], dim=1)

        x_poi = self.relu(self.poi_network2(x_poi_new) + x_poi)
        x_point = self.relu(self.point_network2(x_point_new) + x_point)
        x_pop = self.relu(self.pop_network2(x_pop_new) + x_pop)

        x_poi_new = torch.concat([x_poi, x_point, x_pop], dim=1)
        x_point_new = torch.concat([x_point, x_pop, x_poi], dim=1)
        x_pop_new = torch.concat([x_pop, x_poi, x_point], dim=1)
        x_poi = self.relu(self.poi_network3(x_poi_new) + self.downsample2(x_poi))
        x_point = self.relu(self.point_network3(x_point_new) + self.downsample2(x_point))
        x_pop = self.relu(self.pop_network3(x_pop_new) + self.downsample2(x_pop))

        return torch.concat([x_poi, x_point, x_pop], dim=1)


class ImageDecoderNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageDecoderNet, self).__init__()
        self.network1 = ResBlock(input_dim, input_dim)
        self.network2 = ResBlock(input_dim, hidden_dim, stride=2, downsample_style=2)
        self.network3 = ResBlock(hidden_dim, hidden_dim)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.network1(x)
        x = self.network2(x)
        x = self.network3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class ImageBackboneNet(nn.Module):
    def __init__(self, output_dim):
        super(ImageBackboneNet, self).__init__()
        # Everything except the last linear layer
        backbone = resnet18(pretrained=True)
        layer_list = list(backbone.children())
        self.first_layer = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        )
        self.features = nn.Sequential(*layer_list[1:-1])
        self.last_layer = nn.Sequential(
            nn.Linear(layer_list[-1].in_features, output_dim)
        )

    def forward(self, x):
        x = self.first_layer(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.last_layer(x)
        return x
