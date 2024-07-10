CUDA_VISIBLE_DEVICES=0,1, python -m torch.distributed.launch --nproc_per_node=2 --master_port=62504 --use_env inclearn \
  --options options/continual_vit_fewshot_base/imagenet_pretrained_vit_my_small_patch16_224.yaml \
  options/data/imagenet1000_1order.yaml \
  --initial-increment 1000 \
  --increment 1 \
  --label vit_mySmall_patch16_224_imagenet_pretrained \
  --save task \
  --workers 8