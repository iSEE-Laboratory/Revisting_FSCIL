import argparse
import importlib
from utils import *
from models.logger import LOGGER as log

MODEL_DIR = None
DATA_DIR = 'data/'
PROJECT = 'base'


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot',
                                 'ft_cos'])  # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos',
                                 'avg_cos'])  # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1)  # lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-num_workers_new', type=int, default=0)
    parser.add_argument('-debug', action='store_true')

    # for param aug
    parser.add_argument('-no_tta', action='store_true', help='not tta')
    parser.add_argument('-branches', type=int, default=4)
    parser.add_argument('-K', type=int, default=2, help='free lunch k')
    parser.add_argument('-exp_name', type=str, default='')
    parser.add_argument('-stage0_chkpt', type=str, default='')  # stage0 model is trained in another repo.

    # for feature mapping
    parser.add_argument('-base_fr', type=int, default=1)
    parser.add_argument('-lr_fm', type=float, default=0.1)
    parser.add_argument('-milestones_fm', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-gamma_fm', type=float, default=0.1)
    parser.add_argument('-batch_size_fm', type=int, default=256)
    parser.add_argument('-epochs_fm', type=int, default=100)
    parser.add_argument('-iter_fm', type=int, default=50)

    parser.add_argument('-dataset_seed', type=int, default=None)
    parser.add_argument('-feature_rectification_rkd', type=str, default='distance')
    parser.add_argument('-lr_fr', type=float, default=0.1)
    parser.add_argument('-milestones_fr', nargs='+', type=int, default=[60, 70, 80, 90])
    parser.add_argument('-gamma_fr', type=float, default=0.1)
    parser.add_argument('-batch_size_fr', type=int, default=256)
    parser.add_argument('-epochs_fr', type=int, default=100)
    parser.add_argument('-iter_fr', type=int, default=50)
    parser.add_argument('-resume_fr', type=str, default=None)

    ## feature rectify
    parser.add_argument('-fr_cos', type=float, default=0.1)
    parser.add_argument('-fr_kl', type=float, default=0)
    parser.add_argument('-fr_rkd', type=float, default=1)
    parser.add_argument('-fr_ce_current', type=float, default=1)
    parser.add_argument('-fr_ce_novel', type=float, default=1)
    parser.add_argument('-fr_ce_global', type=float, default=1)
    parser.add_argument('-rkd_inter', type=float, default=0.)
    parser.add_argument('-rkd_intra', type=float, default=0.)
    parser.add_argument('-rkd_intra_extra', type=float, default=0.)
    parser.add_argument('-rkd_inter_extra', type=float, default=0.)

    parser.add_argument('-p', type=int, default=64)
    parser.add_argument('-k', type=int, default=8)

    parser.add_argument('-output_blocks', nargs='+', required=True, default=8, type=int)
    parser.add_argument('-image_size', default=32, type=int)
    parser.add_argument('-re_cal_rectified_center', default=False, type=bool)
    parser.add_argument('-rkd_split', type=str, default='intraI_interI')
    parser.add_argument('-extra_rkd_split', type=str, default='')
    parser.add_argument('-re_extract_avg', type=bool, default=False)
    parser.add_argument('-eps', type=float, default=1e-6)


    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
