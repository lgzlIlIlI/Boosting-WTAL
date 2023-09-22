import numpy as np
import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import model
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool
import pickle
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)


        
class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        embed_dim = 1024
        
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.7),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.7),
                                       nn.Sigmoid())

    def forward(self,vfeat,ffeat):
        filter_feat = vfeat
        x_atn = self.attention(filter_feat)
        return x_atn,filter_feat,ffeat,vfeat
    
class BWA_fusion_dropout_feat_v3(torch.nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        embed_dim = 1024
        self.AE_e = nn.Sequential(
            nn.Conv1d(n_feature, 1024, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5) )
        self.AE_d = nn.Sequential(
            nn.Conv1d( 1024,n_feature, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5) )
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(1024, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
            
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Dropout(0.7),
                                       nn.Conv1d(512, 512, 3, padding=1), nn.LeakyReLU(0.2), 
                                       nn.Conv1d(512, 1, 1), nn.Dropout(0.7),
                                       nn.Sigmoid())
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
    def forward(self,vfeat,ffeat):
        
        fusion_feat = self.AE_e(ffeat)
        new_feat = self.AE_d(fusion_feat)
        
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)#b,1024,1
        vfeat = torch.sigmoid(channel_attn) * vfeat
        channel_attn_norm = channel_attn/torch.norm(channel_attn,p=2,dim=1,keepdim=True)
        bit_wise_attn = self.bit_wise_attn(fusion_feat) #b,1024,320
        bit_wise_attn_norm = bit_wise_attn/torch.norm(bit_wise_attn,p=2,dim=1,keepdim=True)
        temp_attn= torch.einsum('bdn,bdt->bnt',[channel_attn_norm,bit_wise_attn_norm])

        #filter_feat = torch.sigmoid(bit_wise_attn*channel_attn)*vfeat
        filter_feat = torch.sigmoid(bit_wise_attn*temp_attn)*vfeat#+vfeat
        x_atn = self.attention(filter_feat)
        return x_atn,filter_feat,new_feat,vfeat
    
    
    


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out




# attention / transformers



class MHA(nn.Module):
    """
    Multi Head Attention with mask
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the input embedding
        n_head,          # number of heads in multi-head self-attention
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # calculate query, key, values for all heads in batch
        # (B, nh * hs, T)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ v #* mask[:, :, :, None].to(v.dtype)) # juzhengchengfa
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) #* mask.to(out.dtype)
        return out#, mask



# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)

#fusion split modal single+ bit_wise_atten dropout+ contrastive + mutual learning +fusion feat(cat)
#------TOP!!!!!!!!!!

def class_encoder():
    classes = np.load(
    '/data/lgz/ActivityNet1.3-Annotations/' + "classlist.npy", allow_pickle=True)
    vocab = torchtext.vocab.pretrained_aliases["glove.42B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors.cuda(), torch.zeros(1, vocab.dim).cuda()],dim=0)
    embedder = torch.nn.Embedding.from_pretrained(vocab.vectors)
    words_feat = torch.zeros((200,5,300))
    for i,sentence in  enumerate(classes):
        word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_feat = embedder(word_idxs)
        #word_feat = torch.mean(word_feat,dim=0,keepdim=True)
        for j in range(word_feat.shape[0]):
            words_feat[i][j] = word_feat[j]
    return words_feat #200,1,300

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024): #512,0,11
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
   
    
class ANT_CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class,n_pro,**args):
        super().__init__()
        embed_dim=2048
        mid_dim=1024
        dropout_ratio=args['opt'].dropout_ratio

        self.vAttn = BWA_fusion_dropout_feat_v3(1024)
        self.fAttn = BWA_fusion_dropout_feat_v3(1024)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature , 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.fusion2 = nn.Sequential(
            nn.Conv1d(embed_dim , embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        #self.classifier = nn.Sequential( nn.Conv1d(embed_dim, n_class+1, 1))
        self.weight = nn.Parameter(torch.randn(201,300,n_pro))
        torch_init.kaiming_uniform_(self.weight)
        self.b_enc = nn.Parameter(torch.zeros(1,5,300))
        self.a_enc = class_encoder()
        
        self.txt_emb =  nn.Sequential(
            nn.Conv1d(300, 2048 , 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.MHA = MHA(n_embd=2048,n_head=2)
        self.mlp = nn.Sequential(
            nn.Conv1d(2048,8192,1),
            nn.GELU(),
            nn.Dropout(0.1,inplace=True),
            nn.Conv1d(8192,2048,1),
            nn.Dropout(0.1,inplace=True)
        )
        self.position = nn.Parameter(0.01 * torch.randn(201,2048,n_pro+5))
        self.word_pos_encoder = SinusoidalPositionalEmbedding(2048, 0, n_pro+5)
        self.drop_path_attn = AffineDropPath(2048,drop_prob=0.1)
        self.drop_path_mlp = AffineDropPath(2048,drop_prob=0.1)
        
        _kernel = ((args['opt'].max_seqlen // args['opt'].t) // 2 * 2 + 1)
        self.pool=nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()
        self.apply(weights_init)
       
    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b,c,n=feat.size()
        v_atn,vfeat,n_rfeat,o_rfeat= self.vAttn(feat[:,:1024,:],feat[:,1024:,:])
        f_atn,ffeat,n_ffeat ,o_ffeat= self.fAttn(feat[:,1024:,:],feat[:,:1024,:])
        x_atn = (f_atn+v_atn)/2
        nfeat = torch.cat((vfeat,ffeat),1)
        
        nfeat_out0 = self.fusion(nfeat)
        nfeat_out = self.fusion2(nfeat_out0) #b,2048,T
        
        cls_enc = torch.cat((self.a_enc.cuda(),self.b_enc),dim=0) #201,1,300
        new_weight = torch.cat((self.weight,cls_enc.transpose(-1,-2)),dim=2)#201,300,num_pro+1

        pos = self.word_pos_encoder(new_weight.transpose(-1,-2))
        pos = pos / (2048 ** 0.5)
        new_weight = self.txt_emb(new_weight)+self.position+ pos.transpose(-1,-2)  #201,2048,num_pro+1
        new_weight= new_weight + self.drop_path_attn(self.MHA(new_weight))
        new_weight= new_weight + self.drop_path_mlp(self.mlp(new_weight)) #201,2048,num_pro
        new_weight = new_weight[:,:,0] #201,2048
        
        x_cls = torch.einsum('bdn,cd->bcn',[nfeat_out,new_weight])#b,201,T
        #x_cls = self.classifier(nfeat_out)

        x_cls=self.pool(x_cls)
        x_atn=self.pool(x_atn)
        f_atn=self.pool(f_atn)
        v_atn=self.pool(v_atn)

        return {'feat':nfeat_out0.transpose(-1, -2), 'cas':x_cls.transpose(-1, -2), 'attn':x_atn.transpose(-1, -2), 'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2),
             'n_rfeat':n_rfeat.transpose(-1,-2),'o_rfeat':o_rfeat.transpose(-1,-2),'n_ffeat':n_ffeat.transpose(-1,-2),'o_ffeat':o_ffeat.transpose(-1,-2)}


