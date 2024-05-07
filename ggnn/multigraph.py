from torch_geometric import nn as gnn
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
import torch
import random
import math
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from ggnn.rgcn import RGCNConv


class GCNSequential(nn.Sequential):
    """docstring for GCNSequential"""

    def __init__(self, *args, **kwargs):
        super(GCNSequential, self).__init__()
        self.args = args
        self.kwargs = kwargs

        super(GCNSequential, self).__init__(*args, **kwargs)

    def forward(self, input, edge_index):
        for module in self._modules.values():
            input = module(input, edge_index)
        return input


def zoneout(prev_h, next_h, rate, training=True):
    """TODO: Docstring for zoneout.

    :prev_h: TODO
    :next_h: TODO

    :p: when p = 1, all new elements should be droped
        when p = 0, all new elements should be maintained

    :returns: TODO

    """
    from torch.nn.functional import dropout
    if training:
        # bernoulli: draw a value 1.
        # p = 1 -> d = 1 -> return prev_h
        # p = 0 -> d = 0 -> return next_h
        # d = torch.zeros_like(next_h).bernoulli_(p)
        # return (1 - d) * next_h + d * prev_h
        next_h = (1 - rate) * dropout(next_h - prev_h, rate) + prev_h
    else:
        next_h = rate * prev_h + (1 - rate) * next_h

    return next_h


class KStepRGCN(nn.Module):
    """docstring for KStepRGCN"""

    def __init__(
            self,
            in_channels,
            out_channels,
            num_relations,
            num_bases,
            K,
            bias,
    ):
        super(KStepRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.K = K
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(in_channels,
                     out_channels,
                     num_relations,
                     num_bases,
                     bias)
        ] + [
            RGCNConv(out_channels,
                     out_channels,
                     num_relations,
                     num_bases,
                     bias) for _ in range(self.K - 1)
        ])

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.K):
            x = self.rgcn_layers[i](x=x,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    edge_norm=None)
            # not final layer, add relu
            if i != self.K - 1:
                x = torch.relu(x)
        return x


class GLSTMCell(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_type=None,
                 dropout_prob=0.0,
                 num_relations=3,
                 num_bases=3,
                 K=1,
                 num_nodes=80,
                 global_fusion=False):
        """初始化GLSTM单元."""
        super(GLSTMCell, self).__init__()
        self.num_chunks = 4  # LSTM有四个组件: 遗忘门、输入门、输出门和单元状态
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_nodes = num_nodes
        self.global_fusion = global_fusion


        # 为输入和隐藏状态定义了两个R-GCN层，适用于LSTM的四个组件
        self.cheb_i = KStepRGCN(in_channels,
                                out_channels * self.num_chunks,
                                num_relations=num_relations,
                                num_bases=num_bases,
                                K=K,
                                bias=False)
        self.cheb_h = KStepRGCN(out_channels,
                                out_channels * self.num_chunks,
                                num_relations=num_relations,
                                num_bases=num_bases,
                                K=K,
                                bias=False)

        # 初始化LSTM的偏置参数
        self.bias_f = Parameter(torch.Tensor(self.out_channels))  # 遗忘门偏置
        self.bias_i = Parameter(torch.Tensor(self.out_channels))  # 输入门偏置
        self.bias_o = Parameter(torch.Tensor(self.out_channels))  # 输出门偏置
        self.bias_c = Parameter(torch.Tensor(self.out_channels))  # 单元状态更新偏置

        self.dropout_prob = dropout_prob
        self.dropout_type = dropout_type

        # 全局融合
        if global_fusion is True:
            self.mlpi = nn.Linear(self.num_nodes * self.in_channels, self.out_channels)
            self.mlph = nn.Linear(self.num_nodes * self.out_channels, self.out_channels)

            # 全局融合对应的偏置参数，适用于LSTM
            self.bias_f_g = Parameter(torch.Tensor(self.out_channels))  # 全局遗忘门偏置
            self.bias_i_g = Parameter(torch.Tensor(self.out_channels))  # 全局输入门偏置
            self.bias_o_g = Parameter(torch.Tensor(self.out_channels))  # 全局输出门偏置
            self.bias_c_g = Parameter(torch.Tensor(self.out_channels))  # 全局单元状态更新偏置

            self.global_i = nn.Linear(out_channels, out_channels * self.num_chunks)
            self.global_h = nn.Linear(out_channels, out_channels * self.num_chunks)
            self.mlpatt = nn.Linear(self.out_channels * 2, self.out_channels)
            self.ln = nn.LayerNorm([self.out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化所有偏置参数为1，适用于LSTM
        init.ones_(self.bias_f)
        init.ones_(self.bias_i)
        init.ones_(self.bias_o)
        init.ones_(self.bias_c)

        if self.global_fusion is True:
            init.ones_(self.bias_f_g)
            init.ones_(self.bias_i_g)
            init.ones_(self.bias_o_g)
            init.ones_(self.bias_c_g)

    def forward(self, inputs, edge_index, edge_attr, hidden=None):
        """TODO: Docstring for forward.

        :inputs: TODO
        :hidden: TODO
        :returns: TODO

        """
        if hidden is None:
            # 对于LSTM，hidden包含两部分：隐藏状态和单元状态，因此需要初始化两个状态
            h_0 = torch.zeros(inputs.size(0), self.out_channels, dtype=inputs.dtype, device=inputs.device)
            c_0 = torch.zeros(inputs.size(0), self.out_channels, dtype=inputs.dtype, device=inputs.device)
            hidden = (h_0, c_0)
        else:
            h_0, c_0 = hidden

        # R-GCN前向传播
        gi = self.cheb_i(inputs, edge_index=edge_index, edge_attr=edge_attr)
        gh = self.cheb_h(h_0, edge_index=edge_index, edge_attr=edge_attr)

        # 在LSTM中，需要将gi和gh沿特征维度分为四个部分，对应于遗忘门、输入门、单元状态和输出门
        i_f, i_i, i_c, i_o = gi.chunk(4, 1)
        h_f, h_i, h_c, h_o = gh.chunk(4, 1)

        # 计算遗忘门、输入门
        forgetgate = torch.sigmoid(i_f + h_f + self.bias_f)
        inputgate = torch.sigmoid(i_i + h_i + self.bias_i)
        # 计算单元状态的候选值
        cellgate = torch.tanh(i_c + h_c + self.bias_c)
        # 更新单元状态
        next_c = forgetgate * c_0 + inputgate * cellgate
        # 计算输出门
        outputgate = torch.sigmoid(i_o + h_o + self.bias_o)
        # 更新隐藏状态
        next_h = outputgate * torch.tanh(next_c)

        # 返回更新后的隐藏状态和单元状态
        next_hidden = (next_h, next_c)

        if self.global_fusion is True:
            # 将输入数据和隐藏状态展平并进行线性变换，得到了全局输入和全局隐藏状态的表示。
            global_input = self.mlpi(inputs.view(-1, self.num_nodes * self.in_channels))
            h_0, c_0 = hidden
            global_hidden_h = self.mlph(h_0.view(-1, self.num_nodes * self.out_channels))
            global_hidden_c = self.mlph(c_0.view(-1, self.num_nodes * self.out_channels))

            # 全局输入和全局隐藏状态经过线性变换并被分块，得到了对应的遗忘门、输入门、输出门和单元状态更新的值。
            i_f_g, i_i_g, i_o_g, i_c_g = self.global_i(global_input).chunk(4, 1)
            h_f_g, h_i_g, h_o_g, h_c_g = self.global_h(global_hidden_h).chunk(4, 1)

            # LSTM计算
            # 计算遗忘门、输入门、输出门
            f_g = torch.sigmoid(i_f_g + h_f_g + self.bias_f_g)
            i_g = torch.sigmoid(i_i_g + h_i_g + self.bias_i_g)
            o_g = torch.sigmoid(i_o_g + h_o_g + self.bias_o_g)
            # 计算单元状态的新候选值
            c_prime_g = torch.tanh(i_c_g + h_c_g + self.bias_c_g)
            # 更新全局单元状态
            next_c_g = f_g * global_hidden_c + i_g * c_prime_g
            # 更新全局隐藏状态
            next_h_g = o_g * torch.tanh(next_c_g)

            next_h_g = next_h_g.unsqueeze(1).repeat(1, self.num_nodes, 1)
            next_c_g = next_c_g.unsqueeze(1).repeat(1, self.num_nodes, 1)

            # 将之前计算的局部隐藏状态与全局输出合并。
            combine_hidden = torch.cat([next_hidden[0].view(-1, self.num_nodes, self.out_channels), next_h_g],
                                       dim=-1)
            combine_hidden = combine_hidden.view(-1, 2 * self.out_channels)

            combine_c = torch.cat([next_hidden[1].view(-1, self.num_nodes, self.out_channels), next_c_g],
                                       dim=-1)
            combine_c = combine_c.view(-1, 2 * self.out_channels)

            # 利用一个全连接层mlpatt和tanh激活函数，对结合的隐藏状态应用注意力机制。
            attention_weighted_hidden = torch.tanh(self.mlpatt(combine_hidden))
            attention_weighted_c = torch.tanh(self.mlpatt(combine_c))

            # 使用注意力加权的隐藏状态更新局部隐藏状态
            next_hidden_h = next_hidden[0].view(-1, self.out_channels) * attention_weighted_hidden
            next_hidden_c = next_hidden[1].view(-1, self.out_channels) * attention_weighted_c
            # 对获得的新的隐藏状态应用层归一化。
            next_hidden_h = self.ln(next_hidden_h).reshape(-1, self.out_channels)
            next_hidden_c = self.ln(next_hidden_c).reshape(-1, self.out_channels)
            # 更新隐藏状态和单元状态，准备返回
            next_hidden = (next_hidden_h, next_hidden_c)

        output = next_hidden

        if self.dropout_type == 'zoneout':
            # 应用Zoneout到隐藏状态h和单元状态c
            h_0, c_0 = hidden  # 前一时间步的隐藏状态和单元状态
            next_h, next_c = next_hidden  # 当前时间步的隐藏状态和单元状态

            next_h = zoneout(prev_h=h_0, next_h=next_h, rate=self.dropout_prob, training=self.training)
            next_c = zoneout(prev_h=c_0, next_h=next_c, rate=self.dropout_prob, training=self.training)

            next_hidden = (next_h, next_c)  # 更新next_hidden为zoneout处理后的状态

        elif self.dropout_type == 'dropout':
            # 应用Dropout到隐藏状态h，通常不对c应用Dropout
            next_h, next_c = next_hidden  # 当前时间步的隐藏状态和单元状态
            next_h = F.dropout(next_h, self.dropout_prob, self.training)

            next_hidden = (next_h, next_c)  # 更新next_hidden为dropout处理后的状态

        return output, next_hidden


class Net(torch.nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()
        self.num_nodes = cfg['model']['num_nodes']
        self.num_output_dim = cfg['model']['output_dim']
        self.num_units = cfg['model']['rnn_units']
        self.num_input_dim = cfg['model']['input_dim']
        self.num_rnn_layers = cfg['model']['num_rnn_layers']
        self.cfg = cfg
        self.seq_len = cfg['model']['seq_len']
        self.horizon = cfg['model']['horizon']
        self.use_curriculum_learning = self.cfg['model'][
            'use_curriculum_learning']
        self.cl_decay_steps = torch.FloatTensor(
            data=[self.cfg['model']['cl_decay_steps']])
        self.use_go = self.cfg['model'].get('use_go', True)
        self.fusion = self.cfg['model'].get('fusion', 'concat')
        self.dropout_type = cfg['model'].get('dropout_type', None)
        self.dropout_prob = cfg['model'].get('dropout_prob', 0.0)
        self.ar_alpha = cfg['model'].get('ar_alpha', 0)
        self.tar_beta = cfg['model'].get('tar_beta', 0)
        self.use_input = cfg['model'].get('use_input', True)
        self.num_relations = cfg['model'].get('num_relations', 3)
        self.K = cfg['model'].get('K', 3)
        self.num_bases = cfg['model'].get('num_bases', 3)
        act = cfg['model'].get('activation', 'relu')
        act_dict = {
            'relu': F.relu,
            'selu': F.selu,
            'relu6': F.relu6,
            'elu': F.elu,
            'celu': F.celu,
            'leaky_relu': F.leaky_relu,
        }
        self.mediate_activation = act_dict[act]
        self.global_fusion = cfg['model'].get('global_fusion', False)

        self.encoder_cells = nn.ModuleList([
                                               GLSTMCell(self.num_input_dim,
                                                         self.num_units,
                                                         self.dropout_type,
                                                         self.dropout_prob,
                                                         self.num_relations,
                                                         num_bases=self.num_bases,
                                                         K=self.K,
                                                         num_nodes=self.num_nodes,
                                                         global_fusion=self.global_fusion),
                                           ] + [
                                               GLSTMCell(self.num_units,
                                                         self.num_units,
                                                         self.dropout_type,
                                                         self.dropout_prob,
                                                         self.num_relations,
                                                         num_bases=self.num_bases,
                                                         K=self.K,
                                                         num_nodes=self.num_nodes,
                                                         global_fusion=self.global_fusion)
                                               for _ in range(self.num_rnn_layers - 1)
                                           ])

        self.decoder_cells = nn.ModuleList([
                                               GLSTMCell(self.num_input_dim,
                                                         self.num_units,
                                                         self.dropout_type,
                                                         self.dropout_prob,
                                                         self.num_relations,
                                                         num_bases=self.num_bases,
                                                         K=self.K,
                                                         num_nodes=self.num_nodes,
                                                         global_fusion=self.global_fusion),
                                           ] + [
                                               GLSTMCell(self.num_units,
                                                         self.num_units,
                                                         self.dropout_type,
                                                         self.dropout_prob,
                                                         self.num_relations,
                                                         self.K,
                                                         num_nodes=self.num_nodes,
                                                         global_fusion=self.global_fusion)
                                               for _ in range(self.num_rnn_layers - 1)
                                           ])

        # 构建输出层
        self.output_type = cfg['model'].get('output_type', 'fc')
        if not self.fusion == 'concat':
            raise NotImplementedError(self.fusion)

        if self.output_type == 'fc':
            self.output_layer = nn.Linear(self.num_units, self.num_output_dim)
        self.global_step = 0

    @staticmethod
    def _compute_sampling_threshold(step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse
        sigmoid.

        :step: TODO
        :k: TODO
        :returns: TODO

        """
        return k / (k + math.exp(step / k))

    @staticmethod
    def inverse_sigmoid_scheduler_sampling(step, k):
        """TODO: Docstring for linear_scheduler_sampling.
        :returns: TODO

        """
        return k / (k + math.exp(step / k))

    def encode(self, sequences, edge_index, edge_attr=None):
        """
        Encodes input into hidden state on one branch for T steps.

        Return: hidden state on one branch.
        """
        hidden_states = [(None, None)] * len(self.encoder_cells)  # 对于LSTM，我们需要存储两个状态：(h, c)

        outputs = []
        for t, batch in enumerate(sequences):
            cur_input = batch.x
            for i, rnn_cell in enumerate(self.encoder_cells):  # 假设已经将GGRUCell更换为GLSTMCell
                cur_h, cur_c = hidden_states[i]  # 分别提取上一个时间步的隐藏状态和单元状态
                if cur_h is None and cur_c is None:
                    hidden = None
                else:
                    hidden = (cur_h, cur_c)
                cur_out, (next_h, next_c) = rnn_cell(inputs=cur_input,
                                                      edge_index=edge_index,
                                                      edge_attr=edge_attr,
                                                      hidden=hidden)  # LSTM单元返回的是一个元组，包含下一个时间步的隐藏状态和单元状态

                # 更新存储的状态
                hidden_states[i] = (next_h, next_c)
                c_out = cur_out[0]
                cur_input = self.mediate_activation(c_out)
            outputs.append(c_out)

        return outputs, hidden_states

    def forward(self, sequences):
        # encoder
        edge_index = sequences[0].edge_index.detach()
        edge_attr = sequences[0].edge_attr.detach()

        outputs, encoder_hiddens = \
            self.encode(sequences, edge_index=edge_index,
                        edge_attr=edge_attr)

        # decoder
        predictions = []
        decoder_hiddens = encoder_hiddens  # copy states
        GO = torch.zeros(decoder_hiddens[0][0].size()[0],  # 这里使用decoder_hiddens[0][0]来获取第一个LSTM单元的隐藏状态h的维度
                         self.num_output_dim,
                         dtype=encoder_hiddens[0][0].dtype,
                         device=encoder_hiddens[0][0].device)
        decoder_input = GO

        for t in range(self.horizon):
            for i, rnn_cell in enumerate(self.decoder_cells):
                cur_h, cur_c = decoder_hiddens[i]  # 分别提取隐藏状态和单元状态
                cur_out, (next_h, next_c) = rnn_cell(inputs=decoder_input,
                                                      edge_index=edge_index,
                                                      edge_attr=edge_attr,
                                                      hidden=(cur_h, cur_c))

                c_out = cur_out[0]

                decoder_hiddens[i] = (next_h, next_c)  # 更新隐藏状态和单元状态
                decoder_input = self.mediate_activation(c_out)

            out = c_out.reshape(-1, self.num_units)
            out = self.output_layer(out).view(-1, self.num_nodes, self.num_output_dim)
            predictions.append(out)

            if self.training and self.use_curriculum_learning:
                c = random.uniform(0, 1)
                T = self._compute_sampling_threshold(self.global_step, self.cl_decay_steps)
                use_truth_sequence = c < T
            else:
                use_truth_sequence = False

            if use_truth_sequence:
                decoder_input = sequences[t].y
            else:
                decoder_input = out.detach().view(-1, self.num_output_dim)
            if not self.use_input:
                decoder_input = GO.detach()

        if self.training:
            self.global_step += 1

        return torch.stack(predictions).transpose(0, 1)
