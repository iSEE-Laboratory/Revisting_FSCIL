LABEL=$1
gpu=$2
echo $LABEL
echo $gpu
python train.py \
  -project my_vit \
  -dataset cub200 \
  -epochs_base 100 \
  -episode_way 15 \
  -episode_shot 1 \
  -low_way 15 \
  -low_shot 1 \
  -lr_base 0.0002 \
  -lrg 0.0002 \
  -lr_new 0.01 \
  -step 20 \
  -gamma 0.5 \
  -gpu "$gpu" \
  -start_session 1 \
  -num_workers 4 \
  -branches 1 \
  -new_mode \
  avg_cos \
  -stage0_chkpt /home/yuming/TTA/src/ParamAug/results/dev/ParameterAugment/202310/week_4/20231028_fewshot_cub200_m0_F_baseline_pure_my_vit_nano_patch16_cos_300_ra4_lr1e-4_new_corr/net_0_task_0.pth \
  -no_tta \
  -epochs_fr 100 \
  -exp_name $LABEL \
  -feature_rectification_rkd distance \
  -fr_cos 0.1 \
  -fr_rkd 0 \
  -fr_ce_current 0 \
  -fr_ce_novel 0.5 \
  -batch_size_fr 1024 \
  -fr_ce_global 0 \
  -fr_kl 1 -output_blocks 8 9 10 \
  -rkd_intra 1 \
  -rkd_inter 1 \
  -p 32 \
  -k 32 \
  -batch_size_base 1024 \
  -rkd_split 'intraI_interI'\
  -eps 1e-5
