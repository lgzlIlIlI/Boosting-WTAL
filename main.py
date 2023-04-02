from __future__ import print_function
import argparse
import os
from syslog import LOG_LOCAL3
import torch
import model
import model2
import multiprocessing as mp
import wsad_dataset

import random
from test import test
from train import train
import options
import numpy as np
from torch.optim import lr_scheduler
from tqdm import tqdm
import shutil
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

   os.environ["CUDA_VISIBLE_DEVICES"]  = '0'
   result_path = './result_test/FINAL_RESULT.txt'
   result_file = open(result_path,'w')
   pool = mp.Pool(5)
   args = options.parser.parse_args()
   seed=args.seed
   setup_seed(seed)
   print('=============seed: {}, pid: {}============='.format(seed,os.getpid()))
   
   device = torch.device("cuda")
   dataset = getattr(wsad_dataset,args.dataset)(args)
   if 'Thumos' in args.dataset_name:
      max_map=[0]*9
   else:
      max_map=[0]*10
   if not os.path.exists('./ckpt/'):
      os.makedirs('./ckpt/')

   model1 = model.TSM(n_feature = dataset.feature_size, n_class = dataset.num_class,n_pro = args.num_pro,opt=args).to(device)
   model0 = model2.VLC(num_pro=args.num_pro2).to(device)
   
   if args.pretrained_ckpt is not None:
      model1.load_state_dict(torch.load(args.pretrained_ckpt))
   
   optimizer = optim.Adam([
      {"params": model1.parameters()}
      ],
      lr=args.lr, weight_decay=args.weight_decay)

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

   for itr in tqdm(range(args.max_iter)):
      loss = train(itr, dataset, args,model1,model0,optimizer,rec_optimizer,rec_lr_scheduler,mask_optimizer,mask_lr_scheduler, device)
      total_loss+=loss
      if itr % args.interval == 0 and not itr == 0:
                     
         print('Iteration: %d, Loss: %.5f' %(itr, total_loss/args.interval))

         total_loss = 0
         iou,dmap,dap = test(itr, dataset, args, model1,device,pool)
         if 'Thumos' in args.dataset_name:
            cond=sum(dmap[2:7])>sum(max_map[2:7])
         else:
            cond=np.mean(dmap)>np.mean(max_map)
         if cond:
            torch.save(model1.state_dict(), './ckpt/Best_model.pkl')
            max_map = dmap
         print('||'.join(['map @ {} = {:.3f} '.format(iou[i],dmap[i]*100) for i in range(len(iou))]),file = result_file,flush=True)
         print('mAP Avg ALL: {:.3f}'.format(sum(dmap)/len(iou)*100),file = result_file,flush=True)
         
         print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],max_map[i]*100) for i in range(len(iou))]),file = result_file,flush=True)
         max_map = np.array(max_map)
         print('mAP Avg 0.1-0.5: {}, mAP Avg 0.3-0.7: {}, mAP Avg ALL: {}'.format(np.mean(max_map[:5])*100,np.mean(max_map[2:7])*100,np.mean(max_map)*100),file = result_file,flush=True)
         print("------------------pid: {}--------------------".format(os.getpid()),file = result_file,flush=True)


      
