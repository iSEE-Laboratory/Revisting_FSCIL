LABEL=$1
gpu=$2
echo $LABEL
echo $gpu
python train.py \
  -project my_vit \
  -dataset mini_imagenet \
  -image_size 96 \
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
  -stage0_chkpt checkpoints/stage0_chkpts/miniImageNet/net_0_task_0.pth \
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
  -fr_kl 0.1 -output_blocks 8 9 10 \
  -rkd_intra 1 \
  -rkd_inter 1 \
  -p 32 \
  -k 32 \
  -batch_size_base 1024 \
  -rkd_split 'intraI_interI'
#  -resume_fr  checkpoint/cifar100/my_vit/ft_cos-avg_cos-data_init-start_1/Epo_100-Lr_0.0002-Step_20-Gam_0.50-Bs_128-Mom_0.90-T_16.00cifar_patch3_kl_concate_pure/session0_feat_rect.pth
