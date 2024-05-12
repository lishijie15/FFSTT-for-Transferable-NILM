import torch
import torch.nn as nn
import torch.nn.functional as F

class meta_Token(nn.Module):
    def __init__(self, seq_len, d_model):
        super(meta_Token, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.token_num = 5
        self.token_idx = [1, 2, 3, 4, 5]
        self.seq_len = seq_len
        self.token_dim = [64, 96, 128, 160, 192]
        self.num_nodes = 100
        self.rnn_units = 32
        self.memory = self.construct_memory()

    def construct_memory(self):  # meta库里总共有5个可供查询的框架，所以对应每个节点也有10种，每个框架输入维度是32
        memory_dict = nn.ParameterDict()
        for i in range(self.token_num):
            memory_dict[f'Memory{i}'] = nn.Parameter(torch.randn(self.token_idx[i], self.token_dim[i]), requires_grad=True)  # (M, d)
            memory_dict[f'Wq{i}'] = nn.Parameter(torch.randn(self.seq_len, self.token_dim[i]), requires_grad=True)  # project to query  Wq用rnn_units表示维度，代表每一隐态都要进行query
            memory_dict[f'We1{i}'] = nn.Parameter(torch.randn(self.seq_len, self.token_idx[i]), requires_grad=True)  # project memory to embedding
            memory_dict[f'We2{i}'] = nn.Parameter(torch.randn(self.seq_len, self.token_idx[i]), requires_grad=True)  # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def forward(self, x, x_mark):
        # x: [Batch Variate Time]
        x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)
        x_embeddings = []
        for i in range(self.token_num):
            embeddings = torch.matmul(x, self.memory[f'Wq{i}'])
            x_embeddings.append(embeddings)

        return x_embeddings, self.token_num

class query_token(nn.Module):
    def __init__(self, seq_len, d_model):
        super(query_token, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.token_num = 5
        self.token_idx = [1, 2, 3, 4, 5]
        self.seq_len = seq_len
        self.token_dim = [64, 96, 128, 160, 192]
        self.num_nodes = 100
        self.rnn_units = 32
        self.memory = self.construct_memory()

    def query_memory(self, h_t: torch.Tensor):
        for i in range(self.token_num):
            query = torch.matmul(h_t, self.memory[f'Wq{i}'])  # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]  # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]]  # B, N, d
        return value, query, pos, neg

    def forward(self, x, x_mark):
        # x: [Batch Variate Time]
        x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)  #E,ET
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]

        h_att, query, pos, neg = self.query_memory(x)
        print(h_att)