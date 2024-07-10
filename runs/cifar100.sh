LABEL=$1
gpu=$2
echo $LABEL
echo $gpu
python train.py \
  -project my_vit \
  -image_size 36 \
  -dataset cifar100 \
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
  -stage0_chkpt checkpoint/cifar100/my_vit_patch3_pure_82.9/net_0_task_0.pth \
  -no_tta \
  -epochs_fr 100 \
  -exp_name $LABEL \
  -feature_rectification_rkd distance \
  -fr_cos 0.1 \
  -fr_rkd 0 \
  -fr_ce_current 0 \
  -fr_ce_novel 2 \
  -batch_size_fr 1024 \
  -fr_ce_global 0 \
  -fr_kl 0.5 -output_blocks 7 8 9 \
  -rkd_intra 1 \
  -rkd_inter 1 \
  -p 32 \
  -k 32 \
  -batch_size_base 1024 \
  -rkd_split 'intraI_interI'
