import torch
import torch.nn.functional as F
import torch.optim as optim
import utils.wsad_utils as utils
import numpy as np
from torch.autograd import Variable
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.eval_detection import ANETdetection
import wsad_dataset
from eval.detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
import multiprocessing as mp
import options
import model
import proposal_methods as PM
import pandas as pd
from collections import defaultdict
import os

torch.set_default_tensor_type('torch.cuda.FloatTensor')
@torch.no_grad()
def test(itr, dataset, args, model, device,pool):
    model.eval()
    done = False
    instance_logits_stack = []
    labels_stack = []
    proposals = []
    results = defaultdict(dict)
    while not done:
        features, labels,vn, done = dataset.load_data(is_training=False)
        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(Variable(features))
            element_logits = outputs['cas']
            results[vn] = {'cas':outputs['cas'],'attn':outputs['attn']}
            proposals.append(getattr(PM, args.proposal_method)(vn,outputs,labels))
            logits=element_logits.squeeze(0)
        tmp = F.softmax(torch.mean(torch.topk(logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        instance_logits_stack.append(tmp)
        labels_stack.append(labels)

    if not os.path.exists('temp'):
        os.mkdir('temp')
    np.save('temp/{}.npy'.format(args.model_name),results)
    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)
    proposals = pd.concat(proposals).reset_index(drop=True)

    #CVPR2020
    if 'Thumos14' in args.dataset_name:
        iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
        dmap_detect.prediction = proposals
        dmap,dap = dmap_detect.evaluate()
    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95]

        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args,subset='val')
        dmap_detect.prediction = proposals
        dmap,dap = dmap_detect.evaluate()
    
    if args.dataset_name == 'Thumos14':
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                labels_stack[i,:] = np.zeros_like(labels_stack[i,:])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print('Classification map %f' %cmap)
    return iou,dmap,dap

if __name__ == '__main__':
    args = options.parser.parse_args()
    device = torch.device("cuda")
    dataset = wsad_dataset.SampleDataset(args)#getattr(wsad_dataset,args.dataset)(args)
    num_pro2  = 14
    itr = 0
    model1 = model.CO2(n_feature =2048, n_class = 20,n_pro = 9,opt=args).to(device)
    model1.load_state_dict(torch.load('./ckpt/Best.pkl'))
    pool = mp.Pool(5)
    
    iou,dmap ,dap= test(num_pro2,itr, dataset, args, model1,device,pool)
    #for jdx in range(9):
    #    print(dap[jdx])
    print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],dmap[i]*100) for i in range(len(iou))]))
    print('mAP Avg 0.1-0.5: {}, mAP Avg 0.3-0.7: {}, mAP Avg ALL: {}'.format(np.mean(dmap[:5])*100,np.mean(dmap[2:7])*100,np.mean(dmap)*100))

    
