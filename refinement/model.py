import torch
import torch.nn as nn
import math

class LinearProbe_Thick(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearProbe_Thick, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class LinearProbe_Juicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearProbe_Juicy, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearProbe, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        return x

class LinearProbe_PerSceneThick(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scene_num = 4):
        super(LinearProbe_PerSceneThick, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.linears = nn.ModuleList([nn.Linear(input_size, input_size) for i in range(scene_num)])
        
    def forward(self, x, scene_sign):
        x1 = self.linears[0](x[scene_sign == 1])
        x2 = self.linears[1](x[scene_sign == 2])
        x3 = self.linears[2](x[scene_sign == 3])
        x4 = self.linears[3](x[scene_sign == 4])
        x = torch.cat((x1, x2, x3, x4), 0)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class LinearProbe_PerScene(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scene_num = 4):
        super(LinearProbe_PerScene, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linears = nn.ModuleList([nn.Linear(input_size, input_size) for i in range(scene_num)])
        
    def forward(self, x, scene_sign):
        x1 = self.linears[0](x[scene_sign == 1])
        x2 = self.linears[1](x[scene_sign == 2])
        x3 = self.linears[2](x[scene_sign == 3])
        x4 = self.linears[3](x[scene_sign == 4])
        x = torch.cat((x1, x2, x3, x4), 0)
        x = self.relu(x)
        x = self.linear1(x)
        return x

class LinearProbe_Glayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, g_size=64, ref=False):
        super(LinearProbe_Glayer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear_g = nn.Linear(output_size, g_size)
        self.ref = ref
        

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        if not self.ref:
            x = self.linear_g(x)
            x = self.relu(x)
        return x
    

class MHA(nn.Module):
    def __init__(self, input_dim=3, feature_dim = 768, num_heads=8):
        super(MHA, self).__init__()
        # self.linear1 = nn.Linear(input_dim, feature_dim)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(input_dim, feature_dim)
        # self.mha = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        """forward of MultiheadAttentionModel

        Args:
            q (torch.Tensor): (batch_size, opt_points_size, 3)
            k (torch.Tensor): (batch_size, field_size, feature_dim)
            v (torch.Tensor): (batch_size, field_size, feature_dim)

        Returns:
            torch.tensor: (batch_size, opt_points_size, feature_dim)
        """

        q = self.linear1(q)
        q = self.relu(q)
        k = self.linear2(k)
        k = self.relu(k)
        attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) , dim=-1)
        return attn_weight @ v
