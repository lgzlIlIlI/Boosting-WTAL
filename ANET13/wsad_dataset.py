from __future__ import print_function
import numpy as np
import utils.wsad_utils as utils
import random
import os
import options
import nltk
import pickle
def get_video_prompt_templates():
    prompts = [
        'one video of the ',    
    ]
    return prompts

class AntSampleDataset:
    def __init__(self, args, mode="both",sampling='random'):
        
        with open('/data/lgz/original/vocab/vocab.pkl', 'rb') as fp:
            vocab = pickle.load(fp)
        
        self.vocab = vocab
        self.keep_vocab = dict()
        for w, _ in vocab['counter'].most_common(8000):
            self.keep_vocab[w] = self.vocab_size
        
        Blowdrying_feat = np.load('/data/lgz/originalANET13/misc/Blow-drying.npy')
        corps_feat = np.load('/data/lgz/originalANET13/misc/corps.npy')
        flauta_feat = np.load('/data/lgz/originalANET13/misc/flauta.npy')
        guitarra_feat = np.load('/data/lgz/originalANET13/misc/guitarra.npy')
        jackolanterns_feat = np.load('/data/lgz/originalANET13/misc/jack-o-lanterns.npy')
        mooping_feat = np.load('/data/lgz/originalANET13/misc/Mooping.npy')
        gargling_feat = np.load('/data/lgz/originalANET13/misc/Gargling.npy')
        plataform_feat = np.load('/data/lgz/originalANET13/misc/Plataform.npy')
        powerbock_feat = np.load('/data/lgz/originalANET13/misc/Powerbocking.npy')
        rockpaperscissors_feat = np.load('/data/lgz/originalANET13/misc/Rock-paper-scissors.npy')
        sandcastles_feat = np.load('/data/lgz/originalANET13/misc/sandcastles.npy')
        snatch_feat = np.load('/data/lgz/originalANET13/misc/Snatch.npy')

        self.vocab['w2id']['blow-drying'] = 12825
        self.vocab['id2vec'].append(Blowdrying_feat)
        self.vocab['w2id']['corps'] = 12826
        self.vocab['id2vec'].append(corps_feat)
        self.vocab['w2id']['flauta'] = 12827
        self.vocab['id2vec'].append(flauta_feat)
        self.vocab['w2id']['guitarra'] = 12828
        self.vocab['id2vec'].append(guitarra_feat)
        self.vocab['w2id']['jack-o-lanterns'] = 12829
        self.vocab['id2vec'].append(jackolanterns_feat)
        self.vocab['w2id']['mooping'] = 12830
        self.vocab['id2vec'].append(mooping_feat)
        self.vocab['w2id']['gargling'] = 12831
        self.vocab['id2vec'].append(gargling_feat)
        self.vocab['w2id']['plataform'] = 12832
        self.vocab['id2vec'].append(plataform_feat)
        self.vocab['w2id']['powerbocking'] = 12833
        self.vocab['id2vec'].append(powerbock_feat)
        self.vocab['w2id']['rock-paper-scissors'] = 12834
        self.vocab['id2vec'].append(rockpaperscissors_feat)
        self.vocab['w2id']['sandcastles'] = 12835
        self.vocab['id2vec'].append(sandcastles_feat)
        self.vocab['w2id']['snatch'] = 12835
        self.vocab['id2vec'].append(snatch_feat)

        self.keep_vocab['blow-drying'] = 8001
        self.keep_vocab['corps'] = 8002
        self.keep_vocab['flauta'] = 8003
        self.keep_vocab['guitarra'] = 8004
        self.keep_vocab['jack-o-lanterns'] = 8005
        self.keep_vocab['mooping'] = 8006
        self.keep_vocab['gargling'] = 8007
        self.keep_vocab['plataform'] = 8008
        self.keep_vocab['powerbocking'] = 8009
        self.keep_vocab['rock-paper-scissors'] = 8010
        self.keep_vocab['sandcastles'] = 8011
        self.keep_vocab['snatch'] = 8012
        
        
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling=sampling
        self.num_segments = args.max_seqlen
        self.feature_size = args.feature_size
        self.path_to_features = '/data/lgz/ActivityNet1.3-I3D-JOINTFeatures.npy'
        self.path_to_annotations = '/data/lgz/ActivityNet1.3-Annotations/'
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segment.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024
        self.filter()
    def filter(self):
        new_testidx = []
        for idx in self.testidx:
            feat = self.features[idx]
            if len(feat)>10:
                new_testidx.append(idx)
        self.testidx = new_testidx

        new_trainidx = []
        for idx in self.trainidx:
            feat = self.features[idx]
            if len(feat)>10:
                new_trainidx.append(idx)
        self.trainidx = new_trainidx
    @property
    def vocab_size(self):
        return len(self.keep_vocab) + 1
    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            #if s.decode("utf-8") == "training":  # Specific to Thumos14
            if s == 'train':
                self.trainidx.append(i)
            #elif s.decode("utf-8") == "validation":
            elif s == 'val':
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                if self.features[i].sum() == 0:
                    continue
                for label in self.labels[i]:
                    #if label == category.decode("utf-8"):
                    if label == category:
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self,num_pro, n_similar=0, is_training=True, similar_size=2):
        if is_training:
            labels = []
            idx = []
            # Load similar pairs
            if n_similar != 0:
                rand_classid = np.random.choice(
                    len(self.classwiseidx), size=n_similar
                )
                for rid in rand_classid:
                    rand_sampleid = np.random.choice(
                        len(self.classwiseidx[rid]),
                        size=similar_size,
                        replace=False,
                    )

                    for k in rand_sampleid:
                        idx.append(self.classwiseidx[rid][k])

            # Load rest pairs
            if self.batch_size - similar_size * n_similar < 0:
                self.batch_size = similar_size * n_similar

            rand_sampleid = np.random.choice(
                len(self.trainidx),
                size=self.batch_size - similar_size * n_similar,
            )

            for r in rand_sampleid:
                idx.append(self.trainidx[r])
            feat = []
            
            words_feat_batch  = []
            words_batch = []
            words_len_batch = []
            words_id_batch = []
            words_weight_batch = []
            
            for i in idx:
                ifeat = self.features[i]
                labs = self.labels[i]
                prompt = 'one video of '
                
                if len(labs) == 3:   
                    for jdx,lab in enumerate(labs):   
                        lab_ = lab
                        if jdx == 0:
                            pseudo_sent = prompt + lab_ + ','
                        elif jdx == 1:
                            pseudo_sent +=  lab_ + 'and'
                        else:
                            pseudo_sent +=  lab_ + '.'
                elif len(labs) == 2:
                    for jdx,lab in enumerate(labs):   
                        lab_ = lab
                        if jdx == 0:
                            pseudo_sent = prompt + lab_ + 'and'
                        elif jdx == 1:
                            pseudo_sent +=  lab_ + '.'
                elif len(labs) == 1:
                    for jdx,lab in enumerate(labs):   
                        lab_ = lab
                        pseudo_sent = prompt + lab_ + '.'
                
                iwords = []
                iweights = []
                iwords = []
                iweights = []
                n_pro = num_pro
                
                i_words_feat = np.zeros([n_pro+1,300])
                i_weights = np.zeros([n_pro])
                i_words_id = np.zeros([n_pro])
                for word ,tag in nltk.pos_tag(nltk.tokenize.word_tokenize(pseudo_sent)):
                    word = word.lower()
                    #if word in ['one','video','of','the']:
                    #    iweights.append(0)
                    #    iwords.append(word)
                    if word in self.keep_vocab:
                        if 'NN' in tag:
                            iweights.append(2)
                        elif 'VB' in tag:
                            iweights.append(2)
                        elif 'VJJ' in tag or 'RB' in tag:
                            iweights.append(2)
                        else:
                            iweights.append(1)
                        iwords.append(word)
                
                iwords_len = len(iwords)
                i_weights[:iwords_len] = iweights
                iwords_id = [self.keep_vocab[w] for w in iwords]
                i_words_id[:iwords_len] = iwords_id
                try:
                    iwords_feat = [self.vocab['id2vec'][self.vocab['w2id'][iwords[0]]].astype(np.float32)]
                except:
                    print("Error",pseudo_sent,iwords,iweights)
                iwords_feat.extend(self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32) for w in iwords)
                iwords_feat = np.asarray(iwords_feat)
                i_words_feat[:iwords_feat.shape[0],:] = iwords_feat

                words_feat_batch.append(i_words_feat)
                words_id_batch.append(i_words_id)
                words_weight_batch.append(i_weights)
                words_len_batch.append(iwords_len)
                words_batch.append(iwords)
            
            
            for i in idx:
                ifeat = self.features[i]
                
                if self.sampling == 'random':
                    sample_idx = self.random_perturb(ifeat.shape[0])
                elif self.sampling == 'uniform':
                    sample_idx = self.uniform_sampling(ifeat.shape[0])
                elif self.sampling == "all":
                    sample_idx = np.arange(ifeat.shape[0])
                else:
                    raise AssertionError('Not supported sampling !')
                ifeat = ifeat[sample_idx]
                feat.append(ifeat)
            feat = np.array(feat)
            labels = np.array([self.labels_multihot[i] for i in idx])
            words_feat_batch = np.array(words_feat_batch)
            words_id_batch = np.array(words_id_batch)
            words_weight_batch = np.array(words_weight_batch)
            words_len_batch = np.array(words_len_batch)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            return feat, labels,rand_sampleid,words_batch,words_feat_batch,words_id_batch,words_weight_batch,words_len_batch

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            # feat = utils.process_feat(feat, normalize=self.normalize)
            # feature = feature[sample_idx]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            return feat, np.array(labs),vn, done
    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)

if __name__ == '__main__':
    args = options.parser.parse_args()
    dt = AntSampleDataset(args)
    features, labels, pairs_id,words_batch,words_feat_batch,words_id_batch,words_weight_batch,words_len_batch = dt.load_data(n_similar=args.num_similar)
    print(features.shape,labels.shape)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    print(type(seq_len))
    for i in range(features.shape[0]):
        print(words_batch[i],len(words_batch[i]))
        print(words_feat_batch[i].shape)
        print(words_id_batch[i],words_id_batch[i].shape)
        print(words_weight_batch[i],words_weight_batch[i].shape)
        print(words_len_batch[i])
        print('================')