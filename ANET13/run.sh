#---------------------------------------------------------------------------------------------------
# THUMOS14 Training
CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --path-dataset path/to/Thumos14 --num-class 20 --use-model CO2  --max-iter 5000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_3552 --seed 3552 --AWM BWA_fusion_dropout_feat_v2

# THUMOS14 Testing
CUDA_VISIBLE_DEVICES=0 python test.py --dataset-name Thumos14reduced --num-class 20  --path-dataset path/to/Thumos14  --use-model CO2 --model-name CO2_3552

#---------------------------------------------------------------------------------------------------
#ActivityNet Training
CUDA_VISIBLE_DEVICES=1 python main.py --k 5  --dataset-name ActivityNet1.2   --num-class 100 --use-model ANT_CO2  --dataset AntSampleDataset --lr 3e-5 --max-seqlen 60 --model-name ANT_CO2_3552 --seed 3552 --max-iter 22000

# AcitivityNet1.2 Testing
CUDA_VISIBLE_DEVICES=0 python main.py --dataset-name ActivityNet1.2 --dataset AntSampleDataset --num-class 100 --path-dataset path/to/ActivityNet1.2 --use-model ANT_CO2 --model-name ANT_CO2_3552 --max-seqlen 60



