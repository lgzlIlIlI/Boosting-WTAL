from cProfile import label
from os import ftruncate
import numpy as np
import torch
import torchtext
import random
import torch.nn as nn
import torch.nn.functional as F
import math
import model
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool

from modules.multihead_attention import MultiheadAttention
from modules.decoder import TransformerDecoder
from modules.encoder import TransformerEncoder
from modules.transformers import Transformer, DualTransformer



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask

class Attn(torch.nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        embed_dim = 1024
        self.AE_e = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim//2, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5) )
        self.AE_d = nn.Sequential(
            nn.Conv1d( embed_dim//2,n_feature, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5) )
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature//2, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),nn.LeakyReLU(0.2), nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1), nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
    def forward(self,vfeat,ffeat):
        
        fusion_feat = self.AE_e(ffeat)
        new_feat = self.AE_d(fusion_feat)
        
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)#b,1024,1
        channel_attn_norm = channel_attn/torch.norm(channel_attn,p=2,dim=1,keepdim=True)
        bit_wise_attn = self.bit_wise_attn(fusion_feat) #b,1024,320
        bit_wise_attn_norm = bit_wise_attn/torch.norm(bit_wise_attn,p=2,dim=1,keepdim=True)
        temp_attn= torch.einsum('bdn,bdt->bnt',[channel_attn_norm,bit_wise_attn_norm])
        filter_feat = torch.sigmoid(bit_wise_attn*temp_attn)*vfeat

        x_atn = self.attention(filter_feat)
        return x_atn,filter_feat,new_feat,vfeat
    
class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

class VLC(nn.Module):
    def __init__(self,num_pro):
        super().__init__()
        self.dropout = 0.1
        self.vocab_size = 8001
        self.use_negative = True
        self.hid_dim = 512
        self.vAttn = Attn(1024)
        self.fAttn = Attn(1024)
        
        self.frame_fc = nn.Linear(2048, self.hid_dim)
        self.word_fc = nn.Linear(300,self.hid_dim)
        self.mask_vec = nn.Parameter(torch.zeros(300).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(300).float(), requires_grad=True)
        self.trans = DualTransformer(d_model = self.hid_dim,num_heads = 4,num_decoder_layers1 = 3,num_decoder_layers2 = 3)
        self.trans_a = DualTransformer(d_model = self.hid_dim,num_heads = 4,num_decoder_layers1 = 1,num_decoder_layers2 = 1)
        self.fc_rec = nn.Linear(self.hid_dim, self.vocab_size)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(self.hid_dim, 0, num_pro+1)

    
    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = l // 3
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            p = weights[i, :l].cpu().numpy()
            p = p/np.sum(p)
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        # exit(0)
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1,masked_words

    
    def _froze_mask_generator(self):
        for name, param in self.named_parameters():
            if 'Attn' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def _froze_reconstructor(self):
        for name, param in self.named_parameters():
            if 'Attn' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def unfroze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz,T,frames_channel = frames_feat.size()
        frames_feat = frames_feat.transpose(-1,-2)
        v_atn,vfeat,n_rfeat,o_rfeat = self.vAttn(frames_feat[:,:1024,:],frames_feat[:,1024:,:])
        f_atn,ffeat,n_ffeat,o_ffeat = self.fAttn(frames_feat[:,1024:,:],frames_feat[:,:1024,:])
        gauss_weight = (f_atn+v_atn)/2
        gauss_weight = gauss_weight.squeeze()
        nfeat = torch.cat((vfeat,ffeat),1)
        nfeat = nfeat.transpose(-1,-2)
        
        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)

        nfeat = F.dropout(nfeat, self.dropout, self.training)
        nfeat = self.frame_fc(nfeat)
        frames_mask = _generate_mask(nfeat, frames_len)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)
        # proposals scoring
        enc_out_a,h_a = self.trans_a(nfeat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        
        words_feat1, masked_words = self._mask_words(words_feat, words_len, weights=weights) 
        words_feat1 = words_feat1 +  words_pos
        words_feat1 = words_feat[:, :-1]
        words_mask1 = words_mask[:, :-1]
        
        # semantic completion
        _, h ,attn_weight = self.trans(nfeat, frames_mask, words_feat1, words_mask1, decoding=2,gauss_weight=gauss_weight, need_weight=True)
        words_logit = self.fc_rec(h)
        if self.use_negative:
            _, hard_neg_h = self.trans(nfeat, frames_mask, words_feat1, words_mask1, decoding=2)
            hard_neg_words_logit = self.fc_rec(hard_neg_h)

            _, easy_neg_h = self.trans(nfeat, frames_mask, words_feat1, words_mask1, decoding=2, gauss_weight=1-gauss_weight)
            easy_neg_words_logit = self.fc_rec(easy_neg_h)
        else:
            hard_neg_words_logit = None
            easy_neg_words_logit = None

        weights = None
        
        return {
            'hard_neg_words_logit': hard_neg_words_logit,
            'easy_neg_words_logit': easy_neg_words_logit,
            'words_logit': words_logit, 
            'words_id': words_id,
            'weights': weights,
            'words_mask': words_mask[:, :-1],
            'gauss_weight': gauss_weight,
            'gauss_weight_v': gauss_weight,#v_atn,
            'gauss_weight_f': gauss_weight,#f_atn,
            'attn_weight': attn_weight,
            'n_rfeat':n_rfeat.transpose(-1, -2), 'o_rfeat':o_rfeat.transpose(-1, -2),'n_ffeat':n_ffeat.transpose(-1, -2), 'o_ffeat':o_ffeat.transpose(-1, -2)
        }
    
    def cal_nll_loss(self,logit, idx, mask, weights=None):
        eps = 0.1
        acc = (logit.max(dim=-1)[1]==idx).float()
        mean_acc = (acc * mask).sum() / mask.sum()
        
        logit = logit.log_softmax(dim=-1)
        #print(type(idx.unsqueeze(-1)))
        nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -logit.sum(dim=-1)
        nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
        if weights is None:
            nll_loss = nll_loss.masked_fill(mask == 0, 0)
            nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
        else:
            nll_loss = (nll_loss * weights).sum(dim=-1)

        return nll_loss.contiguous(), mean_acc
    
    def rec_loss(self,words_logit, words_id, words_mask, hard_neg_words_logit=None, **kwargs):
        bsz = words_logit.size(0)
        nll_loss, acc = self.cal_nll_loss(words_logit, words_id, words_mask)
        final_loss = nll_loss.mean()

        if hard_neg_words_logit is not None:
            neg_nll_loss, neg_acc = self.cal_nll_loss(hard_neg_words_logit, words_id, words_mask) 
            final_loss = final_loss + neg_nll_loss.mean()
            
        loss_dict = {
            'final_loss': final_loss.item(),
            'nll_loss': nll_loss.mean().item(),
        }
        if hard_neg_words_logit is not None:
            loss_dict.update({
                'neg_nll_loss': neg_nll_loss.mean().item(),
                })

        return final_loss, loss_dict
    

    def ivc_loss(self,words_logit, words_id, words_mask, hard_neg_words_logit=None, easy_neg_words_logit=None, **kwargs):
        bsz = words_logit.size(0)
        nll_loss, acc = self.cal_nll_loss(words_logit, words_id, words_mask)

        if hard_neg_words_logit is not None:
            hard_neg_nll_loss, hard_neg_acc = self.cal_nll_loss(hard_neg_words_logit, words_id, words_mask)
            tmp_0 = torch.zeros_like(nll_loss).to(words_logit.device)
            tmp_0.requires_grad = False
            hard_neg_loss = torch.max(nll_loss - hard_neg_nll_loss + 0.1, tmp_0)
            loss = hard_neg_loss.mean()
        else:
            loss = nll_loss.mean()
        
        if easy_neg_words_logit is not None:
            easy_neg_nll_loss, easy_neg_acc = self.cal_nll_loss(easy_neg_words_logit, words_id, words_mask)
            tmp_0 = torch.zeros_like(nll_loss).to(words_logit.device)
            tmp_0.requires_grad = False
            easy_neg_loss = torch.max(nll_loss - easy_neg_nll_loss + 0.15, tmp_0) #"beta_2": 0.15,
            loss = loss + easy_neg_loss.mean()

        return loss, {
            'ivc_loss': loss.item(),
            'easy_neg_loss':  easy_neg_loss.mean().item() if easy_neg_words_logit is not None else 0.0,
            'hard_neg_loss': hard_neg_loss.mean().item() if hard_neg_words_logit is not None else 0.0,
        }
            