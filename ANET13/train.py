import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from torch.autograd import Variable
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')
def _multiply(x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

def criterion(outputs, labels, **args):
        feat, element_logits, element_atn= outputs['feat'],outputs['cas'],outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        element_logits_supp = _multiply(element_logits, element_atn,include_min=True)
        
        loss_mil_orig, _ = topkloss(element_logits, labels, is_back=True, rat=args['opt'].k,  reduce=None)
        loss_mil_supp, _ = topkloss(element_logits_supp, labels, is_back=False, rat=args['opt'].k,  reduce=None)
        
        loss_3_supp_Contrastive = Contrastive(feat,element_logits_supp,labels,is_back=False)

        loss_norm = element_atn.mean()
        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()
        v_loss_norm = v_atn.mean()
        v_loss_guide = (1 - v_atn -element_logits.softmax(-1)[..., [-1]]).abs().mean()
        f_loss_norm = f_atn.mean()
        f_loss_guide = (1 - f_atn -element_logits.softmax(-1)[..., [-1]]).abs().mean()
        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() +
                      args['opt'].alpha3*loss_3_supp_Contrastive+
                      args['opt'].alpha1*(loss_norm+v_loss_norm+f_loss_norm)/3 +
                      args['opt'].alpha2*(loss_guide+v_loss_guide+f_loss_guide)/3)
        
        return total_loss

def topkloss(element_logits,labels,is_back=True,lab_rand=None,rat=8,reduce=None):    
        if is_back:
            labels_with_back = torch.cat( (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(  (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(element_logits, k=max(1, int(element_logits.shape[-2] // rat)), dim=-2)
        instance_logits = torch.mean( topk_val, dim=-2)
        labels_with_back = labels_with_back / ( torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
                 milloss = milloss.mean()
        return milloss, topk_ind

def Contrastive(x,element_logits,labels,is_back=False):
        if is_back:
            labels = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3*2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i+1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)      # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss

def train(itr,alpha,numpro2, th_b,dataset, args,model1, model0, optimizer, rec_optimizer,rec_lr_scheduler,mask_optimizer,mask_lr_scheduler, device):
#def train(itr,alpha,numpro2, th_b,dataset, args, model1,optimizer, device):
    model1.train()
    features, labels, pairs_id,words_batch,words_feat_batch,words_id_batch,words_weight_batch,words_len_batch = dataset.load_data(num_pro = numpro2,n_similar=args.num_similar)
    
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    frames_len = torch.from_numpy(seq_len).int().to(device)
    words_feat = torch.from_numpy(words_feat_batch).float().to(device)
    words_len = torch.from_numpy(words_len_batch).int().to(device)
    words_id = torch.from_numpy(words_id_batch).long().to(device)
    words_weights = torch.from_numpy(words_weight_batch).float().to(device)
    #interative
    
    outputs = model1(features,seq_len=seq_len,is_training=True,itr=itr,opt=args)
    tal_attn = outputs['attn']
    n_rfeat,o_rfeat,n_ffeat,o_ffeat = outputs['n_rfeat'],outputs['o_rfeat'],outputs['n_ffeat'],outputs['o_ffeat']
    total_loss = criterion(outputs,labels,seq_len=seq_len,device=device,opt=args,itr=itr,pairs_id=pairs_id,inputs=features)
    total_loss = total_loss + th_b*(F.mse_loss(n_rfeat,o_rfeat) + F.mse_loss(n_ffeat,o_ffeat) )
    
    #interative
    model0._froze_mask_generator()
    rec_optimizer.zero_grad()
    outputs0 = model0(features, frames_len, words_id, words_feat, words_len, words_weights)
    rec_attn = outputs0['gauss_weight'].unsqueeze(-1)
    mutual_loss_rec = alpha * F.mse_loss(tal_attn,rec_attn.detach())

    loss0, loss_dict0 = model0.rec_loss(**outputs0)
    total_loss = total_loss + mutual_loss_rec 
    loss0.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model0.parameters(), 10)
    rec_optimizer.step()
    
    model0._froze_reconstructor()
    mask_optimizer.zero_grad()
    outputs00 = model0(features, frames_len, words_id, words_feat, words_len, words_weights)
    n_rfeat1,o_rfeat1,n_ffeat1,o_ffeat1 = outputs00['n_rfeat'],outputs00['o_rfeat'],outputs00['n_ffeat'],outputs00['o_ffeat']
    rec_attn2 = outputs00['gauss_weight'].unsqueeze(-1)
    mutual_loss_rec2 = alpha * F.mse_loss(rec_attn2,tal_attn.detach()) 
    
    ivcloss00, ivc_loss_dict00 = model0.ivc_loss(**outputs00)
    ivcloss00 =ivcloss00 + th_b*(F.mse_loss(n_rfeat1,o_rfeat1) + F.mse_loss(n_ffeat1,o_ffeat1) )
    loss00 = ivcloss00 + mutual_loss_rec2 
    loss_dict0.update(ivc_loss_dict00)
    loss00.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model0.parameters(), 10)
    mask_optimizer.step()
    curr_lr = rec_lr_scheduler.step_update(itr)
    mask_lr_scheduler.step_update(itr)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.data.cpu().numpy()

