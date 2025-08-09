
backbone=$1
vocab=/mnt/e/challenge/github/REG_histgen/dataset/vocab_reg_all.json


if [ "$backbone" = "optimus1" ]; then
    in_dim=1536
    vocab=/mnt/e/challenge/github/REG_histgen/dataset/vocab_reg_${backbone}_all.json
elif [ "$backbone" = "uni2" ]; then
    in_dim=1536
else
    echo "Error: Unsupported backbone '$backbone'"
    exit 1
fi


model='histgen'
max_length=100
epochs=40
region_size=96
prototype_num=512

python main_test_AllinOne.py \
    --image_dir /mnt/d/REG2025/features_test2/${backbone}/pt_files \
    --ann_path /mnt/e/challenge/github/REG_histgen/dataset/test2_${backbone}.json \
    --vocab_path  ${vocab} \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --threshold 10 \
    --batch_size 1 \
    --epochs $epochs \
    --step_size 1 \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --region_size $region_size \
    --prototype_num $prototype_num \
    --save_dir /mnt/e/challenge/github/REG_histgen/results/${backbone}_all \
    --step_size 1 \
    --gamma 0.8 \
    --seed 42 \
    --log_period 1000 \
    --load /mnt/e/challenge/github/REG_histgen/results/${backbone}_all/model_best.pth \
    --beam_size 3 \
    --d_vf ${in_dim}

python process_json.py --csv_path /mnt/e/challenge/github/REG_histgen/results/${backbone}_all/gen_vs_gt.csv