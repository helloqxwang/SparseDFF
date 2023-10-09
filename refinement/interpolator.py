import torch
import numpy as np
from data_loader import MHA_Dataset
from torch.utils.data import Dataset, DataLoader

class home_made_feature_interpolator:

    def __init__(self, points:np.ndarray, features:np.ndarray, device = None) -> None:
        """Initialize the interpolator

        Args:
            points (np.ndarray): (n, 3)
            features (np.ndarray): (n, dim)
            device (_type_, optional): the device used for torch. Defaults to None(auto-detect).
        """
        if device:
            self.dev = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.dev = torch.device('cuda:0')
            else:
                self.dev = torch.device('cpu')
        self.sigma = 0.01
        self.points = torch.from_numpy(points).to(torch.float32).to(self.dev)
        print('points_mean:', self.points.mean(dim=0))
        # self.points = self.points - self.points.mean(dim=0)
        self.features = torch.from_numpy(features).to(torch.float32).to(self.dev)
    
    def get_points(self)->np.ndarray:
        return self.points.cpu().numpy()
    
    def predict(self, query_points:torch.Tensor)->torch.Tensor:
        """
            Get the features of the query points
        params:
            query_points (torch.Tensor): coordinates of the query points(batch_size, num_query_points, dim)
        
        return:
            interpolated_features (torch.Tensor): interpolated features of the query points (batch_size, num_query_points, dim)
        """
        # from pdb import set_trace; set_trace()
        dim = self.points.shape[1]
        b, n, _ = query_points.shape
        query_points = query_points.reshape(-1, dim)
        num_points = self.points.shape[0]
        num_query_points = query_points.shape[0]
        # show_pc(self.points.cpu().detach().numpy(), query_points.cpu().detach().numpy())
        
        # 扩展points和query_points，使其shape变为(num_query_points, num_points, dim)
        # points_exp = self.points.unsqueeze(0).repeat((num_query_points, 1, 1))
        # query_points_exp = query_points.unsqueeze(1).repeat((1, num_points, 1))
        points_exp = self.points[None, :, :]
        query_points_exp = query_points[:, None, :]
        
        # 计算query_points和points之间的欧氏距离，shape为(num_query_points, num_points)
        # dist min max 0.2 0.6
        dists = torch.norm((points_exp - query_points_exp), dim=-1)
        if dists.isnan().any():
            raise ValueError('nan in dists')

        # from torch.distributions.normal import Normal
        # gau = Normal(0, self.sigma)
        # weights = gau.log_prob(dists)
        weights = 1 / (dists + 1e-10)**2
        if weights.isnan().any():
            raise ValueError('nan in weights')
        
        # 对每个query_point，计算其特征的加权平均，shape为(num_query_points, num_features)
        # print(weights.dtype, self.features.dtype)
        if self.features.isnan().any():
            raise ValueError('nan in self.features')
        interpolated_features = torch.mm(weights, self.features) / torch.sum(weights, dim=1, keepdim=True)
        if interpolated_features.isnan().any():
            raise ValueError('nan in interpolated_features')
        
        return interpolated_features.reshape(b, n, -1)

# MHA Try
import torch.nn as nn

# Define the model
class MHA(nn.Module):
    def __init__(self, input_dim=3, feature_dim = 768, num_heads=4):
        super(MHA, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, kdim=input_dim, vdim=feature_dim, batch_first=True)
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        """forward of MultiheadAttentionModel

        Args:
            q (torch.Tensor): (batch_size, opt_points_size, 3)
            k (torch.Tensor): (batch_size, field_size, feature_dim)
            v (torch.Tensor): (batch_size, field_size, feature_dim)

        Returns:
            torch.tensor: (batch_size, opt_points_size, feature_dim)
        """
        x = self.multihead_attn(q, k, v, need_weights=False)
        return x


model = MHA()
dataset = MHA_Dataset(root_dir='/home/user/wangqx/stanford/data', 
                      extrinsic_dir='/home/user/wangqx/stanford/kinect/workspace/calibration.json', 
                      scale=4)
dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None, shuffle=True, num_workers=4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()


# def train(model, optimizer, criterion, train_loader, num_epochs):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, data in enumerate(train_loader, 0):
#             # Get the inputs
#             q, k, v, y = data
#             # Zero the parameter gradients
#             optimizer.zero_grad()
#             # Forward + backward + optimize
#             outputs = model(q, k, v)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()

#             # Print statistics
#             running_loss += loss.item()
#             if i % 100 == 99:    # Print every 100 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 100))
#                 running_loss = 0.0


