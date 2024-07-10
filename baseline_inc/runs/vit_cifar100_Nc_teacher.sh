CUDA_VISIBLE_DEVICES=0,1, python -m torch.distributed.launch --nproc_per_node=2 --master_port=62504 --use_env inclearn \
  --options options/continual_vit_fewshot_base/cifar100_m0_F_baseline_vit_mysmall_patch4_seq_cos_300_purevit_cifar32.yaml \
  options/data/cifar100_fewshot_1orders.yaml \
  --initial-increment 60 \
  --increment 5 \
  --label fewshot_CIFAR100_m0_F_baseline_vit_small_patch4_cos_300_ra4_purevit \
  --save task \
  --workers 4
