import argparse


def get_parser():
    # yapf: disable

    parser = argparse.ArgumentParser("IncLearner",
                                     description="Incremental Learning trainer.")

    # Model related:
    parser.add_argument("-m", "--model", default="icarl", type=str,
                        help="Incremental learner to train.")
    parser.add_argument("-c", "--backbones", default="rebuffi", type=str,
                        help="Backbone backbones.")
    parser.add_argument("--trans_model", default="deit_tiny_patch16_224", type=str,
                        help="Transformer Backbone Type.")
    parser.add_argument("--dropout", default=0., type=float,
                        help="Dropout value.")
    parser.add_argument("-he", "--herding", default=None, type=str,
                        help="Method to gather previous tasks' examples.")
    parser.add_argument("--memory-size", default=2000, type=int,
                        help="Max number of storable examplars.")
    parser.add_argument("-temp", "--temperature", default=1, type=int,
                        help="Temperature used to soften the predictions.")
    parser.add_argument("-fixed-memory", "--fixed-memory", action="store_true",
                        help="Instead of shrinking the memory, it's already at minimum.")
    parser.add_argument("--force_from_stage0", action="store_true", default=False)
    parser.add_argument("--only_stage0", action="store_true", default=False)
    # transformer settings
    parser.add_argument('--input-size', default=96, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--immediate_layer', default=9, type=int, help='layers to output for APG, '
                                                                       'and shadow layers will be frozen')

    # Data related:
    parser.add_argument("-d", "--dataset", default="cifar100", type=str,
                        help="Dataset to test on.")
    parser.add_argument("-inc", "--increment", default=10, type=int,
                        help="Number of class to add per task.")
    parser.add_argument("-b", "--batch-size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-w", "--workers", default=4, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("-t", "--threads", default=1, type=int,
                        help="Number of threads allocated for PyTorch.")
    parser.add_argument("-v", "--validation", default=0., type=float,
                        help="Validation split (0. <= x < 1.).")
    parser.add_argument("-random", "--random-classes", action="store_true", default=False,
                        help="Randomize classes order of increment")
    parser.add_argument("-order", "--order",
                        help="List of classes ordering, to be given in options.")
    parser.add_argument("-max-task", "--max-task", default=None, type=int,
                        help="Cap the number of tasks.")
    parser.add_argument("-onehot", "--onehot", action="store_true",
                        help="Return data in onehot format.")
    parser.add_argument("-initial-increment", "--initial-increment", default=None, type=int,
                        help="Initial increment, may be bigger.")
    parser.add_argument("-sampler", "--sampler",
                        help="Elements sampler.")
    parser.add_argument("--data-path", default="dataset", type=str)

    # Training related:
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument("-wd", "--weight-decay", default=0.05, type=float,
                        help="Weight decay.")
    parser.add_argument("-sc", "--scheduling", default=[49, 63], nargs="*", type=int,
                        help="Epoch step where to reduce the learning rate.")
    parser.add_argument("-lr-decay", "--lr-decay", default=1 / 5, type=float,
                        help="LR multiplied by it.")
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # optimizer
    parser.add_argument("--opt", "--optimizer", default="adamw", type=str,
                        help="Optimizer to use.")
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument("-e", "--epochs", default=150, type=int,
                        help="Number of epochs per task.")

    # Misc:
    parser.add_argument("--label", type=str,
                        help="Experience name, if used a log will be created.")
    parser.add_argument("--autolabel", action="store_true",
                        help="Auto create label based on options files.")
    parser.add_argument("-seed", "--seed", default=[1], type=int, nargs="+",
                        help="Random seed.")
    parser.add_argument("-seed-range", "--seed-range", type=int, nargs=2,
                        help="Seed range going from first number to second (both included).")
    parser.add_argument("-options", "--options", nargs="+",
                        help="A list of options files.")
    parser.add_argument("-save", "--save-model", choices=["never", "last", "task", "first"],
                        default="never",
                        help="Save the network, either the `last` one or"
                             " each `task`'s ones.")
    parser.add_argument("--dump-predictions", default=False, action="store_true",
                        help="Dump the predictions and their ground-truth on disk.")
    parser.add_argument("-log", "--logging", choices=["critical", "warning", "info", "debug"],
                        default="info", help="Logging level")
    parser.add_argument("-resume", "--resume", default=None,
                        help="Resume from previously saved model, "
                             "must be in the format `*_task_[0-9]+\.pth`.")
    parser.add_argument("--resume-first", action="store_true", default=False)
    parser.add_argument("--recompute-meta", action="store_true", default=False)
    parser.add_argument("--no-benchmark", action="store_true", default=False)
    parser.add_argument("--detect-anomaly", action="store_true", default=False)

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                 "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default=['cuda'],
                        help='device to use for training / testing')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    # parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--debug', action="store_true", default=False)

    return parser
