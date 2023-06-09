# Boosting Weakly supervised Temporal Action Localization via Text Information (CVPR2023)
This is the code of our CVPR2023 paper "Boosting Weakly supervised Temporal Action Localization via Text Information"  [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Boosting_Weakly-Supervised_Temporal_Action_Localization_With_Text_Information_CVPR_2023_paper.pdf) <br />

Guozhang Li,De Cheng,Xinpeng Ding, Nannan Wang, Xiaoyu Wang, Xinbo Gao.

# Abstract
Due to the lack of temporal annotation, current Weakly supervised Temporal Action Localization (WTAL) methods are generally stuck into over-complete or incomplete localization. In this paper, we aim to leverage the text information to boost WTAL from two aspects, i.e., (a) the discriminative objective to enlarge the inter-class difference, thus reducing the over-complete; (b) the generative objective to enhance the intra-class integrity, thus finding more complete temporal boundaries. For the discriminative objective, we propose a Text-Segment Mining (TSM) mechanism, which constructs a text description based on the action class label, and regards the text as the query to mine all class-related segments. Without the temporal annotation of actions, TSM compares the text query with the entire videos across the dataset to mine the best matching segments while ignoring irrelevant ones. Due to the shared sub-actions in different categories of videos, merely applying TSM is too strict to neglect the semantic-related segments, which results in incomplete localization. We further introduce a generative objective named Video-text Language Completion (VLC), which focuses on all semantic-related segments from videos to complete the text sentence. We achieve the state-of-the-art performance on THUMOS14 and ActivityNet1.3. Surprisingly, we also find our proposed method can be  seamlessly applied to existing methods, and improve their performances with a clear margin.

# Requirements
GPU: GeForce RTX 3090 \
CUDA: 11.3 \
Python: 3.8.13 \
Pytorch: 1.8.0

# Dataset
We use the 2048-d features provided by in [CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net). The features of THUMOS14 can be obtained from [here](https://drive.google.com/file/d/1SFEsQNLsG8vgBbqx056L9fjA4TzVZQEu/view?usp=sharing).

# Training
```
./bash run.sh
```

# Citation
If you find the code useful in your research, please cite:
```
@InProceedings{Li_2023_CVPR,
    author    = {Li, Guozhang and Cheng, De and Ding, Xinpeng and Wang, Nannan and Wang, Xiaoyu and Gao, Xinbo},
    title     = {Boosting Weakly-Supervised Temporal Action Localization With Text Information},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {10648-10657}
}
```

# Ackonwledgement
This repo contains modified codes from:
[CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net) \
[CNM](https://github.com/minghangz/cnm) \
[SCN](https://github.com/ikuinen/semantic_completion_network) \
We sincerely thank the owners of all these great repos!
