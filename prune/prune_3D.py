import torch
import numpy as np
import os
import cv2
from prune.tools import mesh2rgbd, get_features, depth2pt, pt_vis

def find_match_3D(points:torch.Tensor, img_sign:torch.Tensor, img_num, dis_threshold=0.005, both_left_right=False, lonely_bonus=True):
    """Calculate the “matchbility" of every point.

    For every points, find neighbors in the `gt_points` near than the `dis_threshold`.
    Record the most-match nerghbors' index. (If a point has no match, record -1)
    If point A has the most-match point B, ***AND*** point B has the most-match point A.
    We call A and B are matched and then save the

    Args:
        points (torch.Tensor): (num, 3)
        img_sign (torch.Tensor): (num, ) (from 1)
        img_num (int): The number of images in total.
        dis_threshold (float, optional): The threshold of the distance. Defaults to 0.005.
    
    Returns:
        select_points: (num' , 3)
        select_index: (num, )
    """
    def near_index(index, img_num):
        return ((index - 2)%img_num) + 1, (index%img_num) + 1
    
    left_match = torch.zeros((img_sign.shape[0] +1), dtype=torch.long).to(points.device)
    left_match[-1] = left_match.shape[0] - 1
    right_match = torch.zeros((img_sign.shape[0] +1), dtype=torch.long).to(points.device)
    right_match[-1] = right_match.shape[0] - 1
    img_sign_expand = torch.cat((img_sign, torch.tensor([img_num + 1]).to(img_sign.device)))
    # from pdb import set_trace; set_trace()

    for index in range(1, img_num+1):
        points_ori = points[img_sign==index]
        # (num0, num)
        dis = torch.cdist(points_ori.unsqueeze(0).to(torch.float32), points.unsqueeze(0).to(torch.float32), p=2).squeeze(0)
        # print(dis.max())
        print(dis.median())
        filter_left = torch.logical_and((dis<dis_threshold), (img_sign==near_index(index, img_num)[0]).unsqueeze(0))
        filter_right = torch.logical_and((dis<dis_threshold), (img_sign==near_index(index, img_num)[1]).unsqueeze(0))
        dis_left = dis.clone()
        dis_left[~filter_left] = 1e8
        dis_right = dis.clone()
        dis_right[~filter_right] = 1e8
        min_left, min_index_left = torch.min(dis_left, dim=1)
        min_right, min_index_right = torch.min(dis_right, dim=1)
        # put lonely points to the trash bin :)
        min_index_left[min_left==1e8] = left_match.shape[0] - 1
        min_index_right[min_right==1e8] = right_match.shape[0] - 1
        left_match[img_sign_expand==index] = min_index_left
        right_match[img_sign_expand==index] = min_index_right

    select = torch.zeros(points.shape[0], dtype=torch.bool).to(points.device)
    for i in range(1, img_num+1):
        i_index = torch.nonzero(img_sign==i).squeeze()
        left_select = right_match[left_match[img_sign_expand==i]] == i_index
        right_select = left_match[right_match[img_sign_expand==i]] == i_index
        lonely_bonus_select = torch.logical_and((left_match[img_sign_expand==i] == img_sign.shape[0]),
                                          (right_match[img_sign_expand==i] == img_sign.shape[0]))
        if both_left_right:
            select_fuse = torch.logical_and(left_select, right_select)
        else:
            select_fuse = torch.logical_or(left_select, right_select)

        if lonely_bonus:
            select[img_sign==i] = torch.logical_or(select_fuse, lonely_bonus_select)
        else:
            select[img_sign==i] = select_fuse
    
    selected_index = torch.nonzero(select).squeeze()
    selected_points = points[selected_index]
    return selected_points, selected_index

def find_match_3D_quotient(points:torch.Tensor, img_sign:torch.Tensor, img_num, dis_threshold=0.005, quotien_threshold=0.8, both_left_right=False, lonely_bonus=True):
    """Calculate the “matchbility" of every point.

    For every points, find neighbors in the `gt_points` near than the `dis_threshold`.
    Record the most-match nerghbors' index. (If a point has no match, record -1)
    If point A has the most-match point B, ***AND*** point B has the most-match point A.
    We call A and B are matched and then save the

    Args:
        points (torch.Tensor): (num, 3)
        img_sign (torch.Tensor): (num, ) (from 1)
        img_num (int): The number of images in total.
        dis_threshold (float, optional): The threshold of the distance. Defaults to 0.005.
    
    Returns:
        select_points: (num' , 3)
        select_index: (num, )
    """
    def near_index(index, img_num):
        return ((index - 2)%img_num) + 1, (index%img_num) + 1
    
    left_match = torch.zeros((img_sign.shape[0] +1), dtype=torch.long).to(points.device)
    left_match[-1] = left_match.shape[0] - 1
    right_match = torch.zeros((img_sign.shape[0] +1), dtype=torch.long).to(points.device)
    right_match[-1] = right_match.shape[0] - 1
    left_quotient = torch.zeros((img_sign.shape[0]), dtype=torch.bool).to(points.device)
    right_quotient = torch.zeros((img_sign.shape[0]), dtype=torch.bool).to(points.device)
    img_sign_expand = torch.cat((img_sign, torch.tensor([img_num + 1]).to(img_sign.device)))
    # from pdb import set_trace; set_trace()

    for index in range(1, img_num+1):
        points_ori = points[img_sign==index]
        # (num0, num)
        dis = torch.cdist(points_ori.unsqueeze(0).to(torch.float32), points.unsqueeze(0).to(torch.float32), p=2).squeeze(0)
        # print(dis.max())
        print(dis.median())
        filter_left = torch.logical_and((dis<dis_threshold), (img_sign==near_index(index, img_num)[0]).unsqueeze(0))
        filter_right = torch.logical_and((dis<dis_threshold), (img_sign==near_index(index, img_num)[1]).unsqueeze(0))
        dis_left = dis.clone()
        dis_left[~filter_left] = 1e8
        dis_right = dis.clone()
        dis_right[~filter_right] = 1e8
        ### they are all tuples
        min_left, min_index_left = torch.topk(dis_left, dim=1, k=2, sorted=True)
        min_right, min_index_right = torch.topk(dis_left, dim=1, k=2, sorted=True)
        # put lonely points to the trash bin :)
        min_index_left[min_left==1e8] = left_match.shape[0] - 1
        min_index_right[min_right==1e8] = right_match.shape[0] - 1
        left_match[img_sign_expand==index] = min_index_left[:, 0]
        right_match[img_sign_expand==index] = min_index_right[:, 0]

        quotient_left = min_left[:, 0] / min_left[:, 1]
        quotient_right =  min_right[:, 0] / min_right[:, 1]
        left_quotient[img_sign==index] = quotient_left < quotien_threshold
        right_quotient[img_sign==index] = quotient_right < quotien_threshold

    select = torch.zeros(points.shape[0], dtype=torch.bool).to(points.device)
    for i in range(1, img_num+1):
        left_select = left_quotient[img_sign==i]
        right_select = right_quotient[img_sign==i]
        lonely_bonus_select = torch.logical_and((left_match[img_sign_expand==i] == img_sign.shape[0]),
                                          (right_match[img_sign_expand==i] == img_sign.shape[0]))
        if both_left_right:
            select_fuse = torch.logical_and(left_select, right_select)
        else:
            select_fuse = torch.logical_or(left_select, right_select)

        if lonely_bonus:
            select[img_sign==i] = torch.logical_or(select_fuse, lonely_bonus_select)
        else:
            select[img_sign==i] = select_fuse
    
    selected_index = torch.nonzero(select).squeeze()
    selected_points = points[selected_index]
    return selected_points, selected_index

def vote_3D(points:torch.Tensor, img_sign:torch.Tensor, img_num, dis_threshold=0.1, selected_num=3000):
 
    ballot_box = torch.zeros((img_sign.shape[0]), dtype=torch.long).to(points.device)

    for index in range(1, img_num+1):
        # calculate the votes per image)
        points_ori = points[img_sign==index]
        # (num0, )
        image_ballot = torch.zeros((points_ori.shape[0]), dtype=torch.long).to(points.device)
        # (num0, num)
        dis = torch.cdist(points_ori.unsqueeze(0).to(torch.float32), points.unsqueeze(0).to(torch.float32), p=2).squeeze(0)
        # (num0, )
        dis_filter = dis > dis_threshold
        dis[dis_filter] = 1e8
        x = torch.arange(points_ori.shape[0]).to(points.device)
        y = torch.nonzero(img_sign==index).squeeze().to(points.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        dis[grid_x, grid_y] = 1e8
        # from pdb import set_trace; set_trace()
        ### (num0, ) (num0, )
        min_dis, min_index = torch.min(dis, dim=1)
        # votes their "closest friend" :)
        # if one guy is really lonely, it will vote a random guy. But it will never happened.
        values = torch.ones_like(min_index).to(points.device)
        ballot_box.index_add_(0, min_index, values)

    # from pdb import set_trace; set_trace()
    sort_index = torch.argsort(ballot_box, descending=True)
    # selected_num = 300
    selected_index = sort_index[:selected_num]
    marker = torch.zeros(points.shape[0], dtype=torch.bool)
    marker[selected_index] = True
    selected_points = points[marker]
    return selected_points, marker

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    key = 0
    mesh_path = '/home/user/wangqx/stanford/bunnyQ_Attack1_{}.obj'.format(key)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    img_num = 4
    scale = 4
    img_size = 1024
    imgs, depths_ori, c2w, K, camera_params = mesh2rgbd([mesh_path], device, num=img_num, img_size=img_size)
    print('#############')
    features = get_features(imgs, scale, key, img_num, img_size=img_size)
    n, h, w, c= features.shape
    depths = torch.from_numpy(np.array([cv2.resize(depth, (h , w), interpolation=cv2.INTER_NEAREST) for depth in depths_ori.cpu().numpy()])).to(device)
    points, batch_sign, filter_depth = depth2pt(depths, camera_params, torch.from_numpy(c2w).to(device).to(torch.float32), xyz_images=False, device=device)
    threshold = 0.005
    points_select, index_select= find_match_3D(points, batch_sign, img_num, dis_threshold=threshold)
    print(points_select.shape)
    pt_vis(points_select.cpu().numpy(), size=threshold)