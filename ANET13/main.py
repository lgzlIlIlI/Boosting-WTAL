from __future__ import print_function
import argparse
import os
import torch
import model
import multiprocessing as mp
import wsad_dataset
import gc
import random
from test import test
from train import train

import options
import numpy as np
from torch.optim import lr_scheduler
from tqdm import tqdm
import shutil
import model2
from optimizers import AdamOptimizer
from optimizers.lr_schedulers import InverseSquareRootSchedule
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

import torch.optim as optim

if __name__ == '__main__':
      args = options.parser.parse_args()
      os.environ["CUDA_VISIBLE_DEVICES"] = "3"
      th_b=0.15
      alpha = 0.25# args.alpha / 100
      numpro2 = 10
      result_path = 'NEW0.01pos_MSE0.25_OURATTDrop0.7_Trans_vidDrop0.5_thb0.15'#+str(alpha)
      result_file = open('/data/lgz/originalANET13/result_1007/'+result_path,'w')
      pool = mp.Pool(5)
      
      # seed = random.randint(1,10000)
      seed=args.seed
      print('=============seed: {}, pid: {}============='.format(seed,os.getpid()), file = result_file, flush=True)
      setup_seed(seed)
      # torch.manual_seed(args.seed)
      device = torch.device("cuda")
      dataset = getattr(wsad_dataset,args.dataset)(args)
      if 'Thumos' in args.dataset_name:
         max_map=[0]*9
      else:
         max_map=[0]*10
      if not os.path.exists('./ckpt/'):
         os.makedirs('./ckpt/')
      print(args,file = result_file, flush=True)
      model1 = getattr(model,args.use_model)(dataset.feature_size, dataset.num_class,n_pro = args.numpro,opt=args).to(device)
      model0 = model2.SCmodel(num_pro=numpro2).to(device)

      optimizer = optim.Adam([
         {"params": model1.parameters()}],lr=args.lr, weight_decay=args.weight_decay)

      model0._froze_mask_generator()
      parameters = list(filter(lambda p: p.requires_grad, model0.parameters()))
      args0 = {"lr": 4e-4,
      "weight_decay": 0,
      "warmup_updates": 400,
      "warmup_init_lr": 1e-7}
      rec_optimizer = AdamOptimizer(args0, parameters)
      rec_lr_scheduler = InverseSquareRootSchedule(args0, rec_optimizer)

      model0._froze_reconstructor()
      parameters = list(filter(lambda p: p.requires_grad, model0.parameters()))
      args0 = {
      "lr": 4e-4,
      "weight_decay": 0,
      "warmup_updates": 400,
      "warmup_init_lr": 1e-7
      }
      mask_optimizer = AdamOptimizer(args0, parameters)
      mask_lr_scheduler = InverseSquareRootSchedule(args0, mask_optimizer)
      
      
      total_loss = 0
      lrs = [args.lr, args.lr/5, args.lr/5/5]
      print(model1,file = result_file)
      
      for itr in tqdm(range(args.max_iter)):
         
         loss = train(itr,alpha,numpro2, th_b,dataset, args, model1,model0,optimizer,rec_optimizer,rec_lr_scheduler,mask_optimizer,mask_lr_scheduler, device)
         #loss = train(itr,alpha,numpro2, th_b,dataset, args, model1,optimizer, device)

         total_loss+=loss
         if itr >= 0:
            if (itr) % args.interval == 0: #and not itr == 0:
               print('Iteration: %d, Loss: %.5f' %(itr, total_loss/args.interval),file = result_file, flush=True)
               total_loss = 0
               torch.save(model1.state_dict(), './ckpt/last_' + args.model_name + '.pkl')
               iou,dmap = test(itr,numpro2, dataset, args,model1,device,pool)
               print('||'.join(['map @ {} = {:.3f} '.format(iou[i],dmap[i]*100) for i in range(len(iou))]),file = result_file, flush=True)
               print('mAP Avg ALL: {:.3f}'.format(sum(dmap)/len(iou)*100),file = result_file, flush=True)
               if 'Thumos' in args.dataset_name:
                  cond=sum(dmap[:7])>sum(max_map[:7])
               else:
                  cond=np.mean(dmap)>np.mean(max_map)
               if cond:
                  torch.save(model1.state_dict(), './ckpt/best_' + args.model_name + '.pkl')
                  max_map = dmap

               print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],max_map[i]*100) for i in range(len(iou))]),file = result_file, flush=True)
               max_map = np.array(max_map)
               print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(max_map[:5])*100,np.mean(max_map[:7])*100,np.mean(max_map)*100),file = result_file, flush=True)
               print("------------------pid: {}--------------------".format(os.getpid()),file = result_file, flush=True)


    
