import torch
import logging
from model import LinearProbe, LinearProbe_Thick, LinearProbe_Juicy, LinearProbe_PerScene, \
LinearProbe_PerSceneThick, LinearProbe_Glayer
from criterion import NCESoftmaxLoss
import argparse
from omegaconf import DictConfig, OmegaConf, open_dict
from data_loader import get_loader


class PointNCELossTrainer:

  def __init__(
      self,
      config,
      data_loader, lr = 1e-4, mode = 'base', key=4):
    
    self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    self.mode = mode
    self.gsize = 64
    if mode == 'base':
      model = LinearProbe(768, 768)
    elif mode == 'thick':
      model = LinearProbe_Thick(768, 768 * 4 , 768)
    elif mode == 'juicy':
      model = LinearProbe_Juicy(768, 748 * 4, 768)
    elif mode == 'per_scene':
      model = LinearProbe_PerScene(768, 768 * 4, 768, scene_num=4)
    elif mode == 'per_scene_thick':
      model = LinearProbe_PerSceneThick(768, 768 * 4, 768, scene_num=4)
    elif mode == 'glayer':
      model = LinearProbe_Glayer(768, 768 * 4, 768, g_size=self.gsize, ref=False)
    else:
      raise NotImplementedError
    self.model = model.to(self.device)
    self.config = config
    self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
    
    self.T = config.nceT
    # self.npos = config.npos

    self.data_loader = data_loader
    self.curr_iter = 0
    self.stat_freq = 12
    self.key = key

  def train(self):

    curr_iter = self.curr_iter
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    
    total_loss = 0
    total_num = 0.0
    epoch_num = self.config.max_iter // len(self.data_loader)
    print('Start Training')

    for epoch in range(epoch_num):
      for i, input_dict in enumerate(self.data_loader):
        # print(curr_iter)
        curr_iter += 1
        # epoch = curr_iter / len(self.data_loader)
        batch_loss = self._train_iter(input_dict)
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
        if curr_iter % 20000 == 0:
          torch.save(self.model.state_dict(), f'{self.mode}_key{self.key}_pie_{self.gsize}_T0.007_'+str(curr_iter)+'.pth')
        if curr_iter % 5000 == 0:
          self.scheduler.step()


  def _train_iter(self, input_dict):
      
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0
      data_time = 0
      # input_dict = data_loader_iter.next()

      N0, N1 = input_dict['input0_P'].shape[0], input_dict['input0_P'].shape[0]
      pos_pairs = input_dict['correspondences'].to(self.device)
      if self.mode == 'per_scene' or self.mode == 'per_scene_thick':
        i, j = input_dict['index']
        F0 = self.model(input_dict['input0_F'].to(self.device), i)
        F1 = self.model(input_dict['input1_F'].to(self.device), j)
      else:
        F0 = self.model(input_dict['input0_F'].to(self.device))
        F1 = self.model(input_dict['input1_F'].to(self.device))

      # sample positive pairs 
      # q_unique are the index of valid match index. count is the number of it.
      q_unique, count = pos_pairs[:, 0].unique(return_counts=True)
      # print('q_unique: ', q_unique.shape)  #[3415] [1542]
      # print('pos_pairs: ', pos_pairs.shape) #[38515, 2] [17231, 2]
      uniform0 = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.device)
      uniform1 = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.device)
      uniform2 = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.device)
      uniform3 = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.device)
      ## off is the offset of the index of the positive pairs
      off0 = torch.floor(uniform0*count).long()
      off1 = torch.floor(uniform1*count).long()
      off2 = torch.floor(uniform2*count).long()
      off3 = torch.floor(uniform3*count).long()

      cums = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(count, dim=0)[0:-1]], dim=0)
      # this is a randomlization to select the random patterner if the query is not unique.
      k_sel0 = pos_pairs[:, 1][off0+cums]
      k_sel1 = pos_pairs[:, 1][off1+cums]
      k_sel2 = pos_pairs[:, 1][off2+cums]
      k_sel3 = pos_pairs[:, 1][off3+cums]



      q = F0[q_unique.long()]
      k0 = F1[k_sel0.long()]
      k1 = F1[k_sel1.long()]
      k2 = F1[k_sel2.long()]
      k3 = F1[k_sel3.long()]

      # prune the over sampled positive pairs
      # if self.npos < q.shape[0]:
      #     sampled_inds = np.random.choice(q.shape[0], self.npos, replace=False)
      #     q = q[sampled_inds]
      #     k = k[sampled_inds]
      npos = q.shape[0] 
      if npos < 1:
        return None

      # pos logit
      logits = torch.mm(q, k0.transpose(1, 0)) # npos by npos
      suplement1 = torch.mm(q, k1.transpose(1, 0))
      suplement2 = torch.mm(q, k2.transpose(1, 0))
      suplement3 = torch.mm(q, k3.transpose(1, 0))
      mask = ~torch.eye(q.shape[0], dtype=torch.bool).to(self.device)
      suplement1 = suplement1.masked_select(mask).reshape(npos, npos-1)
      suplement2 = suplement2.masked_select(mask).reshape(npos, npos-1)
      suplement3 = suplement3.masked_select(mask).reshape(npos, npos-1)
      logits = torch.cat([logits, suplement1, suplement2, suplement3], dim=-1)
      # print(logits.shape)
      # compute the L2 loss L1 loss is too costly
      # logits = torch.cdist(q.unsqueeze(0).to(torch.float32), k.unsqueeze(0).to(torch.float32), p=2).squeeze(0)
      labels = torch.arange(npos).cuda().long()
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
  argparser.add_argument('--key', type=int, default=4)
  argparser.add_argument('--norm', type=bool, default=False)
  args = argparser.parse_args()
  base_conf = OmegaConf.load(args.config)
  # base_conf = OmegaConf.load('./config.yaml')
  cli_conf = OmegaConf.from_cli()
  conf = OmegaConf.merge(base_conf, cli_conf)
  data_loader = get_loader(key=args.key, norm=args.norm)
  if args.mode == 'base':
    trainer = PointNCELossTrainer(config=conf, data_loader=data_loader, lr= 1e-5, mode = args.mode, key=args.key)
  elif args.mode == 'thick':
    trainer = PointNCELossTrainer(config=conf, data_loader=data_loader, lr= 1e-4, mode = args.mode, key=args.key)
  elif args.mode == 'juicy':
    trainer = PointNCELossTrainer(config=conf, data_loader=data_loader, lr= 1e-4, mode = args.mode, key=args.key)
  elif args.mode == 'per_scene' or args.mode == 'per_scene_thick':
    trainer = PointNCELossTrainer(config=conf, data_loader=data_loader, lr= 1e-4, mode = args.mode, key=args.key)
  elif args.mode == 'glayer':
    trainer = PointNCELossTrainer(config=conf, data_loader=data_loader, lr= 1e-4, mode = args.mode, key=args.key)
  trainer.train()