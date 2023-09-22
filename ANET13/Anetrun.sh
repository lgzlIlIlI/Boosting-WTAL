alpha=5
for i in {1..5}
do
    python main.py --max-seqlen 60 --lr 3e-5 --k 5 --dataset-name ActivityNet1.3  --num-class 200 --use-model ANT_CO2  --max-iter 50001  --dataset AntSampleDataset --model-name CO2_3552 --seed 3552 --AWM BWA_fusion_dropout_feat_v3 --alpha $alpha
    let alpha=alpha+10
done