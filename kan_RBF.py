import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim


class RBFKANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            num_centers=5,  # RBF中心点数量
            rbf_type='gaussian',  # RBF类型：'gaussian', 'multiquadric', 'inverse_quadratic', 'thin_plate'
            scale_noise=0.1,
            scale_base=1.0,
            scale_rbf=1.0,
            enable_standalone_scale_rbf=True,
            base_activation=torch.nn.SiLU,
            center_range=[-1, 1],
            learn_centers=True,  # 是否学习中心点位置
            learn_width=True,  # 是否学习宽度参数
    ):
        super(RBFKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.rbf_type = rbf_type
        self.learn_centers = learn_centers
        self.learn_width = learn_width

        centers_init = torch.linspace(center_range[0], center_range[1], num_centers)
        
        if learn_centers:
            self.centers = torch.nn.Parameter(
                centers_init.unsqueeze(0).repeat(in_features, 1).clone() 
            )
        else:
            self.register_buffer("centers", centers_init.unsqueeze(0).repeat(in_features, 1))

        # 初始化宽度参数（每个RBF中心一个宽度）
        if learn_width:
            self.log_widths = torch.nn.Parameter(
                torch.zeros(in_features, num_centers)
            )
        else:
            h = (center_range[1] - center_range[0]) / max(num_centers - 1, 1)
            self.register_buffer("log_widths", torch.ones(in_features, num_centers) * math.log(1.0 / h) if h > 0 else 0)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.rbf_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, num_centers)
        )
        
        if enable_standalone_scale_rbf:
            self.rbf_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_rbf = scale_rbf
        self.enable_standalone_scale_rbf = enable_standalone_scale_rbf
        self.base_activation = base_activation()
        self.center_range = center_range

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        with torch.no_grad():
            # 初始化RBF权重
            noise = (
                (torch.rand(self.out_features, self.in_features, self.num_centers) - 0.5)
                * self.scale_noise / max(self.num_centers, 1)
            )
            self.rbf_weight.data.copy_(noise * self.scale_rbf)
            
            if self.learn_centers:
                # 重新初始化中心点，确保在范围内均匀分布
                centers_init = torch.linspace(self.center_range[0], self.center_range[1], self.num_centers)
                self.centers.data.copy_(centers_init.unsqueeze(0).repeat(self.in_features, 1).clone())
            
            if self.learn_width:
                # 初始化宽度，基于中心点间距
                h = (self.center_range[1] - self.center_range[0]) / max(self.num_centers - 1, 1)
                if h > 0:
                    self.log_widths.data.zero_() 
                
            if self.enable_standalone_scale_rbf:
                torch.nn.init.kaiming_uniform_(self.rbf_scaler, a=math.sqrt(5) * self.scale_rbf)

    @property
    def widths(self):
        return torch.exp(self.log_widths)

    def rbf_function(self, x: torch.Tensor):
        """
        计算径向基函数值
        
        Args:
            x (torch.Tensor): 输入张量，形状(batch_size, in_features)
            
        Returns:
            torch.Tensor: RBF基函数值，形状(batch_size, in_features, num_centers)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1) 
        centers_expanded = self.centers.unsqueeze(0) 
        widths_expanded = self.widths.unsqueeze(0) 
        distances = torch.abs(x_expanded - centers_expanded) 
        
        # 根据RBF类型计算函数值
        if self.rbf_type == 'gaussian':
            # 高斯RBF: exp(-(distance/width)^2)
            rbf_values = torch.exp(-(distances * widths_expanded) ** 2)
            
        elif self.rbf_type == 'multiquadric':
            # 多重二次RBF: sqrt(1 + (distance/width)^2)
            rbf_values = torch.sqrt(1 + (distances * widths_expanded) ** 2)
            
        elif self.rbf_type == 'inverse_quadratic':
            # 逆二次RBF: 1 / (1 + (distance/width)^2)
            rbf_values = 1.0 / (1 + (distances * widths_expanded) ** 2)
            
        elif self.rbf_type == 'thin_plate':
            # 薄板样条RBF: distance^2 * log(distance)，处理distance=0的情况
            epsilon = 1e-8
            distances_safe = torch.clamp(distances, min=epsilon)
            rbf_values = distances_safe ** 2 * torch.log(distances_safe)
            
        elif self.rbf_type == 'laplacian':
            # 拉普拉斯RBF: exp(-|distance|/width)
            rbf_values = torch.exp(-distances * widths_expanded)
            
        elif self.rbf_type == 'cauchy':
            # 柯西RBF: 1 / (1 + (distance/width)^2)
            rbf_values = 1.0 / (1 + (distances * widths_expanded) ** 2)
            
        else:
            raise ValueError(f"Unsupported RBF type: {self.rbf_type}")
        
        return rbf_values

    @property
    def scaled_rbf_weight(self):
        if self.enable_standalone_scale_rbf:
            return self.rbf_weight * self.rbf_scaler.unsqueeze(-1)
        return self.rbf_weight

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        rbf_bases = self.rbf_function(x) 
        batch_size = x.size(0)
        
        rbf_flat = rbf_bases.view(batch_size, -1) 
        rbf_weight_flat = self.scaled_rbf_weight.view(self.out_features, -1) 
        rbf_output = F.linear(rbf_flat, rbf_weight_flat)
        
        return base_output + rbf_output

    def regularization_loss(self, 
                          regularize_activation=1.0, 
                          regularize_entropy=1.0, 
                          center_penalty=0.01,
                          width_penalty=0.01):
        """
        计算RBF-KAN的正则化损失
        
        Args:
            regularize_activation: RBF权重的L1正则化权重
            regularize_entropy: 熵正则化权重
            center_penalty: 中心点分布均匀性惩罚
            width_penalty: 宽度参数平滑性惩罚
            
        Returns:
            torch.Tensor: 正则化损失
        """
        # RBF权重的L1正则化
        l1_fake = self.scaled_rbf_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()

        p = l1_fake / (regularization_loss_activation + 1e-8)
        regularization_loss_entropy = -torch.sum(p * torch.log(p + 1e-8))
        
        # 中心点分布均匀性惩罚
        if self.learn_centers:
            center_loss = 0
            for i in range(self.in_features):
                centers_sorted, _ = torch.sort(self.centers[i])
                gaps = centers_sorted[1:] - centers_sorted[:-1]
                center_loss += torch.exp(-gaps).mean()
            center_loss = center_loss / self.in_features * center_penalty
        else:
            center_loss = 0
            
        # 宽度参数平滑性惩罚
        if self.learn_width:
            width_loss = 0
            widths = self.widths
            for i in range(self.in_features):
                width_diff = widths[i, 1:] - widths[i, :-1]
                width_loss += (width_diff ** 2).mean()
            width_loss = width_loss / self.in_features * width_penalty
        else:
            width_loss = 0
            
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
            + center_loss
            + width_loss
        )


class RBFKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            num_centers=5,
            rbf_type='gaussian',
            scale_noise=0.1,
            scale_base=1.0,
            scale_rbf=1.0,
            base_activation=torch.nn.SiLU,
            center_range=[-1, 1],
            learn_centers=True,
            learn_width=True,
    ):
        super(RBFKAN, self).__init__()
        self.num_centers = num_centers
        self.rbf_type = rbf_type

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                RBFKANLinear(
                    in_features,
                    out_features,
                    num_centers=num_centers,
                    rbf_type=rbf_type,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_rbf=scale_rbf,
                    base_activation=base_activation,
                    center_range=center_range,
                    learn_centers=learn_centers,
                    learn_width=learn_width,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, 
                          regularize_activation=1.0, 
                          regularize_entropy=1.0,
                          center_penalty=0.01,
                          width_penalty=0.01):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy, center_penalty, width_penalty)
            for layer in self.layers
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class MultiHeadRBFKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            heads=1,
            num_centers=5,
            rbf_type='gaussian',
            scale_noise=0.1,
            scale_base=1.0,
            scale_rbf=1.0,
            base_activation=torch.nn.SiLU,
            center_range=[-1, 1],
            learn_centers=True,
            learn_width=True,
    ):
        super(MultiHeadRBFKAN, self).__init__()
        self.heads = heads
        self.rbf_kans = torch.nn.ModuleList([
            RBFKAN(layers_hidden, num_centers, rbf_type, scale_noise, scale_base, 
                  scale_rbf, base_activation, center_range, learn_centers, learn_width) 
            for _ in range(heads)
        ])

    def forward(self, x: torch.Tensor):
        output = torch.stack([kan(x[:, i, :]) for i, kan in enumerate(self.rbf_kans)], dim=1)
        return output

    def eforward(self, x: torch.Tensor):
        outputs = [kan(x) for kan in self.rbf_kans]
        return torch.stack(outputs, dim=1)

    def reset_parameters(self):
        for kan in self.rbf_kans:
            kan.reset_parameters()


def make_rbf_kans(input_dim, hidden_dim, output_dim, heads, hidden_layers, num_centers, rbf_type='gaussian'):
    sizes = [input_dim] + [hidden_dim] * (hidden_layers - 2) + [output_dim]
    return MultiHeadRBFKAN(layers_hidden=sizes, heads=heads, num_centers=num_centers, rbf_type=rbf_type)


class SparseRBFKAAGCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_centers=5, rbf_type='gaussian', heads=1, concat=True):
        super(SparseRBFKAAGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat

        self.attention_rbf = RBFKANLinear(
            in_features=2 * in_features,
            out_features=heads,
            num_centers=num_centers,
            rbf_type=rbf_type,
        )

        self.linear = torch.nn.Linear(in_features, out_features * heads)

        self.self_param = torch.nn.Parameter(torch.Tensor(out_features * heads))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        torch.nn.init.ones_(self.self_param)

    def forward(self, x, edge_index):
        h = self.linear(x)
        h = h.view(-1, self.heads, self.out_features) 

        row, col = edge_index
        x_i = x[row] 
        x_j = x[col] 

        pair_features = torch.cat([x_i, x_j], dim=1) 
        attention_scores = self.attention_rbf(pair_features)  
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
        attention_weights = self.sparse_softmax(attention_scores, edge_index, x.size(0))

        neighbor_agg = self.sparse_aggregation(attention_weights, h, edge_index)
        output = neighbor_agg + self.self_param.view(1, self.heads, self.out_features) * h

        if self.concat and self.heads > 1:
            output = output.view(-1, self.heads * self.out_features)
        else:
            output = output.mean(dim=1)

        return F.elu(output) if self.concat else output

    def sparse_softmax(self, attention_scores, edge_index, num_nodes):
        """Sparse softmax normalization"""
        row, col = edge_index

        max_per_node = torch.zeros(num_nodes, self.heads,
                                   device=attention_scores.device)
        max_per_node.scatter_reduce_(0, col.unsqueeze(1).expand(-1, self.heads),
                                     attention_scores, reduce='amax', include_self=False)

        exp_scores = torch.exp(attention_scores - max_per_node[col])
        sum_exp = torch.zeros(num_nodes, self.heads, device=attention_scores.device)
        sum_exp.scatter_add_(0, col.unsqueeze(1).expand(-1, self.heads), exp_scores)

        attention_weights = exp_scores / (sum_exp[col] + 1e-8)

        return attention_weights

    def sparse_aggregation(self, attention_weights, h, edge_index):
        """Sparse attention aggregation"""
        row, col = edge_index
        num_nodes = h.size(0)

        source_features = h[row]

        weighted_features = attention_weights.unsqueeze(-1) * source_features  

        aggregated = torch.zeros(num_nodes, self.heads, self.out_features,
                                 device=h.device)
        aggregated.scatter_add_(0, col.unsqueeze(1).unsqueeze(2).expand(-1, self.heads, self.out_features),
                                weighted_features)

        return aggregated

    def update(self, adj_matrix, vectors, layer_idx=0):
        """Message passing using RBF KAA-GCN layer"""

        edge_index = self.dense_to_sparse(adj_matrix)
        return self(vectors, edge_index)

    def dense_to_sparse(self, adj_tensor):
        """Convert dense adjacency matrix to sparse edge indices"""

        edge_index = adj_tensor.nonzero(as_tuple=False).t()
        return edge_index




class ImprovedRBFKAAGCN(torch.nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output, num_centers=5, 
                 rbf_type='gaussian', heads=4, dropout_rate=0.1,
                 learn_centers=True, learn_width=True):
        super(ImprovedRBFKAAGCN, self).__init__()
        self.embed_fingerprint = nn.Embedding(N, dim)

        self.dropout = nn.Dropout(dropout_rate)

        # RBF KAGCN
        self.rbf_kaa_layers = nn.ModuleList()
        for i in range(layer_hidden):
            in_dim = dim if i == 0 else (dim * heads if heads > 1 else dim)
            out_dim = dim
            self.rbf_kaa_layers.append(
                SparseRBFKAAGCNLayer(in_dim, out_dim, num_centers, rbf_type, heads)
            )

        output_dim = dim * heads if heads > 1 else dim
        self.W_output = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim), 
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(layer_output)
        ])
        self.W_property = nn.Linear(output_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def pad(self, matrices, pad_value):
        """Pad the list of matrices for batch processing."""
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def sum(self, vectors, axis):
        """Sum vectors along the given axis."""
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def gnn(self, inputs):
        Smiles, fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        fingerprint_vectors = self.dropout(fingerprint_vectors)

        for i, layer in enumerate(self.rbf_kaa_layers):
            fingerprint_vectors = layer.update(adjacencies, fingerprint_vectors, i)
            if i < len(self.rbf_kaa_layers) - 1: 
                fingerprint_vectors = self.dropout(fingerprint_vectors)

        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return Smiles, molecular_vectors

    def mlp(self, vectors):
        for layer in self.W_output:
            vectors = layer(vectors)
        outputs = self.W_property(vectors)
        return outputs

    def forward_regressor(self, data_batch, train):
        """Forward pass - regressor"""
        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            Smiles, molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                Smiles, molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return Smiles, predicted_values, correct_values


KANLinear = RBFKANLinear
ImprovedKAAGCN = ImprovedRBFKAAGCN
SparseKAAGCNLayer = SparseRBFKAAGCNLayer


"""class MultiModalFusion(nn.Module):
    def __init__(self, gnn_dim, bert_dim, fusion_dim, fusion_type='attention'):
        super(MultiModalFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
            self.bert_proj = nn.Linear(bert_dim, fusion_dim)
            self.attention = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),  
                nn.Tanh(),                           
                nn.Linear(fusion_dim, 2),       
                nn.Softmax(dim=-1)                    
            )
            self.fusion_layer = nn.Linear(fusion_dim, fusion_dim)
    
    def forward(self, gnn_features, bert_features):
        if self.fusion_type == 'attention':
            gnn_proj = self.gnn_proj(gnn_features)   
            bert_proj = self.bert_proj(bert_features)  
            
            concat_features = torch.cat([gnn_proj, bert_proj], dim=-1) 
            attention_weights = self.attention(concat_features) 
            stacked = torch.stack([gnn_proj, bert_proj], dim=1) 
            fused = torch.sum(stacked * attention_weights.unsqueeze(-1), dim=1)

            return self.fusion_layer(fused)
        """

class MultiModalFusion(nn.Module):
    def __init__(self, gnn_dim, bert_dim, fusion_dim, fusion_type='concat', num_heads=4):
        super(MultiModalFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(gnn_dim + bert_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif fusion_type == 'attention':
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
            self.bert_proj = nn.Linear(bert_dim, fusion_dim)
            self.attention = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Tanh(),
                nn.Linear(fusion_dim, 2),
                nn.Softmax(dim=-1)
            )
            self.fusion_layer = nn.Linear(fusion_dim, fusion_dim)
        elif fusion_type == 'gated':
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
            self.bert_proj = nn.Linear(bert_dim, fusion_dim)
            self.gate = nn.Sequential(
                nn.Linear(gnn_dim + bert_dim, fusion_dim),
                nn.Sigmoid()
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU()
            )
        elif fusion_type == 'bilinear':
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
            self.bert_proj = nn.Linear(bert_dim, fusion_dim)
            self.bilinear = nn.Bilinear(fusion_dim, fusion_dim, fusion_dim)
            self.fusion_layer = nn.Sequential(
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif fusion_type == 'cross_attention':
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
            self.bert_proj = nn.Linear(bert_dim, fusion_dim)
            self.num_heads = num_heads
            self.head_dim = fusion_dim // num_heads
            
            assert fusion_dim % num_heads == 0, "fusion_dim must be divisible by num_heads"
            
            self.q_proj_gnn = nn.Linear(fusion_dim, fusion_dim)
            self.k_proj_bert = nn.Linear(fusion_dim, fusion_dim)
            self.v_proj_bert = nn.Linear(fusion_dim, fusion_dim)
            
            self.q_proj_bert = nn.Linear(fusion_dim, fusion_dim)
            self.k_proj_gnn = nn.Linear(fusion_dim, fusion_dim)
            self.v_proj_gnn = nn.Linear(fusion_dim, fusion_dim)
            
            self.out_proj = nn.Linear(fusion_dim * 2, fusion_dim)
            self.layer_norm = nn.LayerNorm(fusion_dim)
            self.dropout = nn.Dropout(0.1)
        elif fusion_type == 'qformer':
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
            self.bert_proj = nn.Linear(bert_dim, fusion_dim)
            
            self.num_queries = 8
            self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, fusion_dim))
            
            self.qformer_layers = nn.ModuleList()
            num_qformer_layers = 2
            for _ in range(num_qformer_layers):
                self.qformer_layers.append(
                    QFormerLayer(fusion_dim, num_heads)
                )
            
            self.output_proj = nn.Sequential(
                nn.Linear(self.num_queries * fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def _cross_attention(self, query, key, value, batch_size):
        query = query.view(batch_size, self.num_heads, self.head_dim)
        key = key.view(batch_size, self.num_heads, self.head_dim)
        value = value.view(batch_size, self.num_heads, self.head_dim)
        
        scores = torch.sum(query * key, dim=-1) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = attn_weights.unsqueeze(-1) * value
        output = output.view(batch_size, -1)
        
        return output
    
    def forward(self, gnn_features, bert_features):
        if self.fusion_type == 'concat':
            combined = torch.cat([gnn_features, bert_features], dim=-1)
            return self.fusion_layer(combined)
        
        elif self.fusion_type == 'attention':
            gnn_proj = self.gnn_proj(gnn_features)
            bert_proj = self.bert_proj(bert_features)
            
            concat_features = torch.cat([gnn_proj, bert_proj], dim=-1)
            attention_weights = self.attention(concat_features)
            
            stacked = torch.stack([gnn_proj, bert_proj], dim=1)
            fused = torch.sum(stacked * attention_weights.unsqueeze(-1), dim=1)
            return self.fusion_layer(fused)
        
        elif self.fusion_type == 'gated':
            gnn_proj = self.gnn_proj(gnn_features)
            bert_proj = self.bert_proj(bert_features)
            
            concat = torch.cat([gnn_features, bert_features], dim=-1)
            gate_values = self.gate(concat)
            
            fused = gate_values * gnn_proj + (1 - gate_values) * bert_proj
            return self.fusion_layer(fused)
        
        elif self.fusion_type == 'bilinear':
            gnn_proj = self.gnn_proj(gnn_features)
            bert_proj = self.bert_proj(bert_features)
            
            fused = self.bilinear(gnn_proj, bert_proj)
            return self.fusion_layer(fused)
        
        elif self.fusion_type == 'cross_attention':
            gnn_proj = self.gnn_proj(gnn_features)
            bert_proj = self.bert_proj(bert_features)
            
            batch_size = gnn_proj.size(0)
            
            q_gnn = self.q_proj_gnn(gnn_proj)
            k_bert = self.k_proj_bert(bert_proj)
            v_bert = self.v_proj_bert(bert_proj)
            
            q_bert = self.q_proj_bert(bert_proj)
            k_gnn = self.k_proj_gnn(gnn_proj)
            v_gnn = self.v_proj_gnn(gnn_proj)
            
            g2b_output = self._cross_attention(q_gnn, k_bert, v_bert, batch_size)
            b2g_output = self._cross_attention(q_bert, k_gnn, v_gnn, batch_size)
            
            fused = torch.cat([g2b_output, b2g_output], dim=-1)
            fused = self.out_proj(fused)
            fused = self.layer_norm(0.5*fused + gnn_proj + bert_proj)
            
            return fused
        
        elif self.fusion_type == 'qformer':
            gnn_proj = self.gnn_proj(gnn_features)
            bert_proj = self.bert_proj(bert_features)
            
            batch_size = gnn_proj.size(0)
            
            query = self.query_tokens.expand(batch_size, -1, -1)
            
            kv = torch.stack([gnn_proj, bert_proj], dim=1)
            
            for layer in self.qformer_layers:
                query = layer(query, kv)
            
            fused = query.reshape(batch_size, -1)
            fused = self.output_proj(fused)
            
            return fused


class QFormerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super(QFormerLayer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (dim + num_heads - 1) // num_heads
        self.effective_dim = self.head_dim * num_heads
        
        self.q_proj = nn.Linear(dim, self.effective_dim)
        self.k_proj = nn.Linear(dim, self.effective_dim)
        self.v_proj = nn.Linear(dim, self.effective_dim)
        self.out_proj = nn.Linear(self.effective_dim, dim)
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1),
        )
        
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, query, key_value):
        """
        query: (B, N_q, D) - 可学习查询向量
        key_value: (B, N_kv, D) - GNN和BERT拼接的键值对
        """
        B = query.size(0)
        
        residual = query
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        Q = self.q_proj(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.effective_dim)
        out = self.out_proj(out)
        
        query = residual + self.gate.tanh() * out
        
        residual = query
        query = self.norm_ffn(query)
        query = residual + self.gate.tanh() * self.ffn(query)
        
        return query


class QFormerProjector(nn.Module):
    def __init__(self, gnn_dim, bert_dim, num_queries=8, num_heads=8, num_layers=2):
        super(QFormerProjector, self).__init__()
        self.num_queries = num_queries
        self.bert_dim = bert_dim
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, bert_dim) * 0.02)
        
        self.gnn_proj = nn.Linear(gnn_dim, bert_dim)
        self.gnn_ln = nn.LayerNorm(bert_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                QFormerProjectorLayer(bert_dim, num_heads)
            )
        
        self.output_ln = nn.LayerNorm(bert_dim)
    
    def forward(self, node_features, molecular_sizes):
        """
        node_features: (total_nodes, D_gnn) - 所有分子的节点特征拼接
        molecular_sizes: list of int - 每个分子的节点数
        
        Returns: (B, num_queries, D_bert) - 每个分子的软提示
        """
        kv = self.gnn_ln(self.gnn_proj(node_features))
        
        batch_query_tokens = []
        offset = 0
        for size in molecular_sizes:
            query = self.query_tokens.clone()
            mol_kv = kv[offset:offset + size].unsqueeze(0)
            
            for layer in self.layers:
                query = layer(query, mol_kv)
            
            batch_query_tokens.append(self.output_ln(query).squeeze(0))
            offset += size
        
        return torch.stack(batch_query_tokens, dim=0)


class QFormerProjectorLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super(QFormerProjectorLayer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (dim + num_heads - 1) // num_heads
        self.effective_dim = self.head_dim * num_heads
        
        self.self_attn_q = nn.Linear(dim, self.effective_dim)
        self.self_attn_k = nn.Linear(dim, self.effective_dim)
        self.self_attn_v = nn.Linear(dim, self.effective_dim)
        self.self_attn_out = nn.Linear(self.effective_dim, dim)
        self.self_attn_ln = nn.LayerNorm(dim)
        self.self_attn_gate = nn.Parameter(torch.zeros(1))
        
        self.cross_attn_q = nn.Linear(dim, self.effective_dim)
        self.cross_attn_k = nn.Linear(dim, self.effective_dim)
        self.cross_attn_v = nn.Linear(dim, self.effective_dim)
        self.cross_attn_out = nn.Linear(self.effective_dim, dim)
        self.cross_attn_ln = nn.LayerNorm(dim)
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn_gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, query, kv):
        """
        query: (1, num_queries, D) - 可学习查询向量
        kv: (1, num_nodes, D) - 分子的节点特征
        
        Returns: (1, num_queries, D)
        """
        B = query.size(0)
        
        # Self-Attention among query tokens
        residual = query
        q_norm = self.self_attn_ln(query)
        Q = self.self_attn_q(q_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.self_attn_k(q_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.self_attn_v(q_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.effective_dim)
        query = residual + self.self_attn_gate.tanh() * self.self_attn_out(out)
        
        # Cross-Attention: query attends to node features
        residual = query
        q_norm = self.cross_attn_ln(query)
        kv_norm = self.cross_attn_ln(kv)
        Q = self.cross_attn_q(q_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.cross_attn_k(kv_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.cross_attn_v(kv_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.effective_dim)
        query = residual + self.cross_attn_gate.tanh() * self.cross_attn_out(out)
        
        # FFN
        residual = query
        query = residual + self.ffn_gate.tanh() * self.ffn(self.ffn_ln(query))
        
        return query


class MultiModalRBFKAAGCN(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output, vocab_size,
                 bert_dim=256, bert_heads=8, bert_layers=4, bert_ff_dim=1024,
                 num_centers=5, rbf_type='gaussian', heads=1, dropout_rate=0.1,
                 fusion_type='attention', fusion_heads=4, learn_centers=True, learn_width=True,
                 max_smiles_length=256):
        super(MultiModalRBFKAAGCN, self).__init__()

        self.embed_fingerprint = nn.Embedding(N, dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.rbf_kaa_layers = nn.ModuleList()
        for i in range(layer_hidden):
            in_dim = dim if i == 0 else (dim * heads if heads > 1 else dim)
            out_dim = dim
            self.rbf_kaa_layers.append(
                SparseRBFKAAGCNLayer(in_dim, out_dim, num_centers, rbf_type, heads)
            )
        
        gnn_output_dim = dim * heads if heads > 1 else dim
        
        from molbert import MolBERT
        self.molbert = MolBERT(
            vocab_size=vocab_size,
            d_model=bert_dim,
            num_heads=bert_heads,
            num_layers=bert_layers,
            d_ff=bert_ff_dim,
            max_length=max_smiles_length,
            dropout=dropout_rate,
            output_dim=bert_dim
        )
        
        self.num_soft_prompt_tokens = 8
        self.qformer_projector = QFormerProjector(
            gnn_dim=gnn_output_dim,
            bert_dim=bert_dim,
            num_queries=self.num_soft_prompt_tokens,
            num_heads=8,
            num_layers=2
        )
        
        fusion_dim = max(gnn_output_dim, bert_dim)
        if fusion_type == 'cross_attention':
            fusion_dim = max(fusion_dim, fusion_heads * 64)
            if fusion_dim % fusion_heads != 0:
                fusion_dim = ((fusion_dim // fusion_heads) + 1) * fusion_heads
        
        self.fusion = MultiModalFusion(
            gnn_dim=gnn_output_dim,
            bert_dim=bert_dim,
            fusion_dim=fusion_dim,
            fusion_type=fusion_type,
            num_heads=fusion_heads
        )
        
        self.W_output = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(layer_output)
        ])
        self.W_property = nn.Linear(fusion_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices
    
    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)
    
    def gnn(self, inputs):
        Smiles, fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        fingerprint_vectors = self.dropout(fingerprint_vectors)

        for i, layer in enumerate(self.rbf_kaa_layers):
            fingerprint_vectors = layer.update(adjacencies, fingerprint_vectors, i)
            if i < len(self.rbf_kaa_layers) - 1:
                fingerprint_vectors = self.dropout(fingerprint_vectors)

        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return Smiles, molecular_vectors, fingerprint_vectors
    
    def encode_smiles(self, smiles_list, input_ids_list, attention_mask_list, soft_prompt=None):
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        
        bert_outputs = self.molbert(input_ids, attention_mask, soft_prompt)
        return bert_outputs['projected_output']
    
    def mlp(self, vectors):
        for layer in self.W_output:
            vectors = layer(vectors)
        outputs = self.W_property(vectors)
        return outputs
    
    def forward_regressor(self, data_batch, train):
        Smiles = data_batch[0]
        fingerprints = data_batch[1]
        adjacencies = data_batch[2]
        molecular_sizes = data_batch[3]
        correct_values = torch.cat(data_batch[4])
        input_ids_list = data_batch[5]
        attention_mask_list = data_batch[6]

        if train:
            _, gnn_features, node_features = self.gnn((Smiles, fingerprints, adjacencies, molecular_sizes))
            
            soft_prompt = self.qformer_projector(node_features, molecular_sizes)
            
            bert_features = self.encode_smiles(Smiles, input_ids_list, attention_mask_list, soft_prompt)

            fused_features = self.fusion(gnn_features, bert_features)

            predicted_values = self.mlp(fused_features)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                _, gnn_features, node_features = self.gnn((Smiles, fingerprints, adjacencies, molecular_sizes))
                
                soft_prompt = self.qformer_projector(node_features, molecular_sizes)
                
                bert_features = self.encode_smiles(Smiles, input_ids_list, attention_mask_list, soft_prompt)

                fused_features = self.fusion(gnn_features, bert_features)

                predicted_values = self.mlp(fused_features)

            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return Smiles, predicted_values, correct_values


class MultiModalRBFKAAGCNWithPretrainedBERT(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output, 
                 pretrained_model_name='seyonec/ChemBERTa-zinc-base-v1',
                 num_centers=5, rbf_type='gaussian', heads=4, dropout_rate=0.1,
                 fusion_type='attention', learn_centers=True, learn_width=True,
                 freeze_bert=False):
        super(MultiModalRBFKAAGCNWithPretrainedBERT, self).__init__()
        
        self.embed_fingerprint = nn.Embedding(N, dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.rbf_kaa_layers = nn.ModuleList()
        for i in range(layer_hidden):
            in_dim = dim if i == 0 else (dim * heads if heads > 1 else dim)
            out_dim = dim
            self.rbf_kaa_layers.append(
                SparseRBFKAAGCNLayer(in_dim, out_dim, num_centers, rbf_type, heads)
            )
        
        gnn_output_dim = dim * heads if heads > 1 else dim
        
        try:
            from transformers import AutoModel, AutoTokenizer
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            bert_dim = self.bert.config.hidden_size
            
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            self.use_pretrained = True
        except ImportError:
            print("transformers not installed, using custom MolBERT")
            self.use_pretrained = False
            from molbert import MolBERT
            bert_dim = 256
            self.bert = MolBERT(
                vocab_size=1000,
                d_model=bert_dim,
                num_heads=8,
                num_layers=4,
                d_ff=1024,
                output_dim=bert_dim
            )
        
        self.num_soft_prompt_tokens = 8
        self.qformer_projector = QFormerProjector(
            gnn_dim=gnn_output_dim,
            bert_dim=bert_dim,
            num_queries=self.num_soft_prompt_tokens,
            num_heads=8,
            num_layers=2
        )
        
        fusion_dim = max(gnn_output_dim, bert_dim)
        self.fusion = MultiModalFusion(
            gnn_dim=gnn_output_dim,
            bert_dim=bert_dim,
            fusion_dim=fusion_dim,
            fusion_type=fusion_type
        )
        
        self.W_output = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(layer_output)
        ])
        self.W_property = nn.Linear(fusion_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices
    
    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)
    
    def gnn(self, inputs):
        Smiles, fingerprints, adjacencies, molecular_sizes = inputs

        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        fingerprint_vectors = self.dropout(fingerprint_vectors)

        for i, layer in enumerate(self.rbf_kaa_layers):
            fingerprint_vectors = layer.update(adjacencies, fingerprint_vectors, i)
            if i < len(self.rbf_kaa_layers) - 1:
                fingerprint_vectors = self.dropout(fingerprint_vectors)

        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return Smiles, molecular_vectors, fingerprint_vectors
    
    def encode_smiles_pretrained(self, smiles_list):
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(next(self.bert.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.bert.parameters()).device)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output
    
    def encode_smiles_custom(self, input_ids_list, attention_mask_list, soft_prompt=None):
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        
        bert_outputs = self.bert(input_ids, attention_mask, soft_prompt)
        return bert_outputs['projected_output']
    
    def mlp(self, vectors):
        for layer in self.W_output:
            vectors = layer(vectors)
        outputs = self.W_property(vectors)
        return outputs
    
    def forward_regressor(self, data_batch, train):
        Smiles = data_batch[0]
        fingerprints = data_batch[1]
        adjacencies = data_batch[2]
        molecular_sizes = data_batch[3]
        correct_values = torch.cat(data_batch[4])
        input_ids_list = data_batch[5] if len(data_batch) > 5 else None
        attention_mask_list = data_batch[6] if len(data_batch) > 6 else None

        if train:
            _, gnn_features, node_features = self.gnn((Smiles, fingerprints, adjacencies, molecular_sizes))
            
            soft_prompt = self.qformer_projector(node_features, molecular_sizes)

            if self.use_pretrained:
                bert_features = self.encode_smiles_pretrained(Smiles)
            else:
                bert_features = self.encode_smiles_custom(input_ids_list, attention_mask_list, soft_prompt)

            fused_features = self.fusion(gnn_features, bert_features)

            predicted_values = self.mlp(fused_features)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                _, gnn_features, node_features = self.gnn((Smiles, fingerprints, adjacencies, molecular_sizes))
                
                soft_prompt = self.qformer_projector(node_features, molecular_sizes)

                if self.use_pretrained:
                    bert_features = self.encode_smiles_pretrained(Smiles)
                else:
                    bert_features = self.encode_smiles_custom(input_ids_list, attention_mask_list, soft_prompt)
                
                fused_features = self.fusion(gnn_features, bert_features)
                
                predicted_values = self.mlp(fused_features)
            
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return Smiles, predicted_values, correct_values