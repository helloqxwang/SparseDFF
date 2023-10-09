import torch
import logging
from model import MHA
from criterion import NCESoftmaxLoss
import argparse
from omegaconf import DictConfig, OmegaConf, open_dict
from data_loader import get_loader_MHA
import numpy as np


class MHA_Trainer:

  def __init__(
      self,
      config,
      data_loader, lr = 1e-3, mode = 'base', key=0):
    
    self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    self.mode = mode
    if mode == 'base':
      model = MHA(input_dim=3, feature_dim=768, num_heads=8)
    else:
      raise NotImplementedError
    self.model = model.to(self.device)
    self.config = config
    self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
    
    # self.npos = config.npos

    self.data_loader = data_loader
    self.curr_iter = 0
    self.stat_freq = 12
    self.key = key
    self.sigma = 3
    self.n_opt_pts = 256
    self.b_size = 8
    self.T = config.nceT

  def train(self):

    curr_iter = self.curr_iter
    
    total_loss = 0
    total_num = 0.0
    epoch_num = self.config.max_iter // len(self.data_loader)
    print('Start Training')

    for epoch in range(epoch_num):
      for i, input_pair in enumerate(self.data_loader):
        # print(curr_iter)
        curr_iter += 1
        # epoch = curr_iter / len(self.data_loader)
        batch_loss = self._train_iter(input_pair)
        if batch_loss is None:
          continue
        total_loss += batch_loss
        total_num += 1

        # if curr_iter % self.lr_update_freq == 0 or curr_iter == 1:
        #   lr = self.scheduler.get_last_lr()
        #   self.scheduler.step()
        #   if self.is_master:
        #     logging.info(f" Epoch: {epoch}, LR: {lr}")
        #     self._save_checkpoint(curr_iter, 'checkpoint_'+str(curr_iter))

        # Print logs
        if curr_iter % self.stat_freq == 0:
          # self.writer.add_scalar('train/loss', batch_loss, curr_iter)
          logging.info(
              "Train Epoch: {:.3f} [{}/{}], Current Loss: {:.3e}"
              .format(epoch, curr_iter,
                      len(self.data_loader), batch_loss) 
          )
          print("Train Epoch: {:.3f} [{}/{}], Current Loss: {:.3e}"
              .format(epoch, curr_iter,
                      len(self.data_loader), batch_loss))
        if curr_iter % 10000 == 0:
          torch.save(self.model.state_dict(), f'mha_{self.mode}_key{self.key}_monkey'+str(curr_iter)+'.pth')
        if curr_iter % 5000 == 0:
          self.scheduler.step()


  def _train_iter(self, input_pair):
      
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0
      data_time = 0
      # input_dict = data_loader_iter.next()

      points, features = input_pair
      ref_points = points[0].to(self.device)
      ref_features = features[0].to(self.device)
      query_points = points[1].to(self.device)
      query_features = features[1].to(self.device)
      sample_pts = torch.from_numpy(np.random.normal(0.0, self.sigma, size=(self.n_opt_pts, 3))).to(self.device)
      ref_offset_ind = torch.randint(0, ref_points.shape[0], (self.b_size,)).to(self.device)

    # compute the anchor points in ref_points
      ref_anchors = ref_points[ref_offset_ind] # (b_size, 3)
      ref_offset = (ref_anchors - ref_points.mean(axis=0)) * 1.05 + ref_points.mean(axis=0)
      ref_q_pts = (sample_pts[None, ...] + ref_offset[:, None, :]).to(torch.float32) # (b_size, n_opt_pts, 3)

    # compute the corresponding anchor points in query_points
      dis = torch.cdist(ref_features[ref_offset_ind].unsqueeze(0), query_features.unsqueeze(0), p=2).squeeze(0) # (b_size, n_query_pts)
      query_offset_ind = torch.argmin(dis, dim=1) # (b_size, )
      query_anchors = query_points[query_offset_ind] # (b_size, 3)
      query_offset = (query_anchors - query_points.mean(axis=0)) * 1.05 + query_points.mean(axis=0)
      query_q_pts = (sample_pts[None, ...] + query_offset[:, None, :]).to(torch.float32) # (b_size, n_opt_pts, 3)
      
    # (b_size, n_opt_pts, feat_dim)
      # from pdb import set_trace; set_trace()
      # print(ref_q_pts[0])
      ref_interpolate_feats = self.model(q=ref_q_pts, k=ref_points[None, ...].expand((self.b_size, -1, -1)), v=ref_features[None, ...].expand((self.b_size, -1, -1)))
      query_interpolate_feats = self.model(q=query_q_pts, k=query_points[None, ...].expand((self.b_size, -1, -1)), v=query_features[None, ...].expand((self.b_size, -1, -1)))
      # print(query_features[query_offset_ind])
      # print('#################')
      # print(query_interpolate_feats[0])


    # we assume that there is a one-to-one correspondence between the ref and query points
      r = ref_interpolate_feats.reshape(-1, ref_interpolate_feats.shape[-1]) # (b_size * n_opt_pts, feat_dim)
      q = query_interpolate_feats.reshape(-1, query_interpolate_feats.shape[-1]) # (b_size * n_opt_pts, feat_dim)

      pt_num = r.shape[0]

      x_idx = torch.arange(self.n_opt_pts).cuda().long().tile((self.n_opt_pts,)).tile((self.b_size,))
      x_offset = (torch.arange(self.b_size).cuda().long() * self.n_opt_pts).repeat_interleave(self.n_opt_pts * self.n_opt_pts)
      x_idx = x_idx + x_offset

      y_idx = torch.arange(self.n_opt_pts).cuda().long().repeat_interleave(self.n_opt_pts).tile((self.b_size))
      y_offset = (torch.arange(self.b_size).cuda().long() * self.n_opt_pts ).repeat_interleave(self.n_opt_pts * self.n_opt_pts)
      y_idx = y_idx + y_offset

      # pos logit
      logits = torch.mm(q, r.transpose(1, 0)) # (b_size * n_opt_pts, b_size * n_opt_pts)
      # compute the L2 loss L1 loss is too costly
      # logits = torch.cdist(q.unsqueeze(0).to(torch.float32), k.unsqueeze(0).to(torch.float32), p=2).squeeze(0)

      masks = masks = torch.ones_like(logits, dtype=torch.bool).cuda()
      masks[y_idx, x_idx] = False

      trace_idx = torch.arange(pt_num).cuda().long()
      masks[trace_idx, trace_idx] = True

      logits = logits.masked_select(masks).reshape(pt_num, pt_num - self.n_opt_pts + 1)
      labels = (torch.arange(self.b_size).cuda().long() * self.n_opt_pts).repeat_interleave(self.n_opt_pts)
      # labels = torch.arange(pt_num).cuda().long()

      out = torch.div(logits, self.T)
      out = out.squeeze().contiguous()
      # from pdb import set_trace; set_trace()

      criterion = NCESoftmaxLoss().cuda()
      loss = criterion(out, labels)
      # print('loss: ', loss)

      loss.backward()

      result = {"loss": loss}
      batch_loss += result["loss"].item()

      self.optimizer.step()

      torch.cuda.empty_cache()
      return batch_loss
  
if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--config', type=str, default='./config.yaml')
  argparser.add_argument('--mode', type=str, default='base')
  argparser.add_argument('--key', type=int, default=0)
  argparser.add_argument('--norm', type=bool, default=False)
  args = argparser.parse_args()
  base_conf = OmegaConf.load(args.config)
  # base_conf = OmegaConf.load('./config.yaml')
  cli_conf = OmegaConf.from_cli()
  conf = OmegaConf.merge(base_conf, cli_conf)
  data_loader = get_loader_MHA(key=args.key, norm=args.norm)
  if args.mode == 'base':
    trainer = MHA_Trainer(config=conf, data_loader=data_loader, lr= 1e-2, mode = args.mode, key=args.key)
  trainer.train()