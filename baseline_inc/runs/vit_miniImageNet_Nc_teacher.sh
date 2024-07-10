CUDA_VISIBLE_DEVICES=0,1, python -m torch.distributed.launch --nproc_per_node=2 --master_port=62504 --use_env inclearn \
  --options options/continual_vit_fewshot_base/miniimagenet_m0_F_baseline_vit_mysmall_patch8_seq_cos_300_ra4_purevit_lrmin1e-6.yaml \
  options/data/miniimagenet_fewshot_1orders.yaml \
  --initial-increment 60 \
  --increment 5 \
  --label fewshot_miniImageNet_m0_F_baseline_vit_small_patch8_seq_cos_300_ra4_purevit_lrmin1e-6 \
  --save task \
  --workers 4
