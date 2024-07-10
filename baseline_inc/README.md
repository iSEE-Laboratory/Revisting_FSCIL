## Base Task Training
This sub-directory is used to train on the base task using distributed data parallel (DDP).

### Step1. Download checkpoints
- #### *mini*Imagenet and CIFAR-100
It is difficult to train the model from scratch to match the first-task performance as the CNN (ResNets). 
So we use a ResNet teacher to assist with the first-task training.
To match the performance of the base task with the SOTA method [Neural Collapse](https://github.com/NeuralCollapseApplications/FSCIL),
we use their checkpoint as a teacher on *mini*imagenet and CIFAR-100.

Please download the teacher checkpoints at these URLs:

[Baidu(password=0000)](https://pan.baidu.com/s/1BiPgFVGRDNN0gYSopTfwFg
), [Google](https://drive.google.com/file/d/16O8c-9s9BpBH_-v_iigHmlrTdDsW4WIi/view?usp=drive_link) or [OneDrive](https://sunyatsen-my.sharepoint.cn/:u:/g/personal/tangym9_ms_sysu_edu_cn/EaxC8tvXq15NrfMgQrzZWO0BFwmzlMtkNdOMoB4OTNqL9Q?e=lmc3q0).

After downloading, please put them in `./techer_chkpts`.

Note: It is optional to use RresNet teachers, but it may decrease the performance w/o such teachers.

- #### CUB200
Following the FSCIL community, experiments on CUB200 are based on the ImageNet-pretrained backbone.
We provided the ImageNet-pretrained checkpoint of our modified ViT which can be downloaded at

[Baidu(password=0000)](https://pan.baidu.com/s/1klPV9IUmK53f054KZKICsA), [Google](https://drive.google.com/file/d/1efyN4f_818nOhXImbN9YzgOMnJNvqwyi/view?usp=sharing) or [OneDrive](https://sunyatsen-my.sharepoint.cn/:u:/g/personal/tangym9_ms_sysu_edu_cn/EaLMO4OTU2JHqtv5Pe81G70BuDnIYHteHP4CdOlgCVf0tQ?e=PlUGoe).

After downloading, please put it as `./pretrained_imageNet_patch16_224/best_checkpoint.pth`.

### Step2. start training
- #### *mini*Imagenet
`bash vit_miniImageNet_Nc_teacher.sh`

- #### cifar100
`bash vit_cifar100_Nc_teacher.sh`

- #### CUB200
`bash baseline_inc/runs/vit_cub200_pretrained.sh` (optional)

`bash baseline_inc/runs/vit_cub200_stage0.sh`

### Step3. Organizing checkpoints.
After getting the checkpoints for the base task which locates in `./baseline_inc/results/dev/...`, please copy the checkpoints `net_0_task_0.pth`
to `./checkpoints/stage0_chkpts/[dataset_name]`.
And then, you can `cd` back to the main directory for training the rest of the tasks (novel tasks).