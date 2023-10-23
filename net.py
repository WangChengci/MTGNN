from layer import *


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length

        # 时序卷积模块：膨胀（空洞）卷积 + inception层
        # 膨胀（空洞）卷积:dilation_exponential > 1
        # inception层:dilated_inception
        # self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
        # 该模块用于处理时间序列，根据不同的膨胀率和层数，来构建卷积层
        kernel_size = 7
        if dilation_exponential>1:
            # 非标准卷积，每个kernel的像素点之间有空隙
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                # 添加卷积层
                # filter_convs和gate_convs??
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    # whether to add graph convolution layer
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            # receptive_field感受野大小，表示这个神经网络模型能接受的输入范围
            # 如果输入self.seq_length < self.receptive_field，那么就会进行pad(填充)操作
            # 使得输入数据的长度达到receptive_field大小，以便模型可以正常运行
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        # 是否使用gcn
        if self.gcn_true:
            # 是否构造邻接矩阵
            # 否则就是用预先定义好的邻接矩阵
            if self.buildA_true:
                # self.idx = torch.arange(self.num_nodes).to(device)  idx为图的节点序列
                # 对图进行图构造(graph learning layer)
                # self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
                if idx is None:
                    # gc(self.idx) return一个自适应的邻接矩阵
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        #  self.start_conv = nn.Conv2d(in_channels=in_dim,
        #                              out_channels=residual_channels,
        #                              kernel_size=(1, 1))
        x = self.start_conv(input)

        # 使用残差连接(Residual connections)和跳跃连接(skip connections)来避免梯度消失问题

        # self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
        # self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
        #             kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        # 什么是跳跃连接
        # Skip connection（跳跃连接）是一种神经网络架构设计的技巧，
        # 它的目的是将某一层的输入直接传递到网络中的后续层，以避免梯度消失或梯度爆炸问题，以及提高网络训练的稳定性和效果。
        # kernel_size=(1, self.seq_length)：这表示卷积核的大小，其中 1 表示在时间维度上的卷积核大小，
        # 而 self.seq_length 表示卷积核在时间上的宽度等于输入数据的长度，这使得它能够跳过整个时间序列的信息。


        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        # 看到这了
        for i in range(self.layers):
            # 时间卷积模块应用一组标准扩张一维卷积滤波器来提取高级时间特征。
            # 该模块由两个扩展的初始层组成。
            # 一个扩张的初始层后面是一个tanh()激活函数，并充当过滤器。
            # 另一层后面是 sigmoid 激活函数，用作控制过滤器可以传递到下一个模块的信息量的门。
            residual = x

            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)

            x = filter * gate

            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
