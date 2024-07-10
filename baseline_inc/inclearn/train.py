import copy
import json
import os
import pickle
import pprint
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml

from inclearn.lib import factory
# from inclearn.lib import logger as logger_lib
from inclearn.lib import metrics, results_utils, utils
# logger = logging.getLogger(__name__)
from inclearn.lib.logger import LOGGER

log = LOGGER
logger = log.LOGGER


def train(args):
    # logger.set_logging_level(args["logging"].upper())

    autolabel = _set_up_options(args)
    if args["autolabel"]:
        args["label"] = autolabel

    if args["label"]:
        logger.info("Label: {}".format(args["label"]))
        try:
            os.system("echo '\ek{}\e\\'".format(args["label"]))
        except:
            pass
    if args["resume"] and not os.path.exists(args["resume"]):
        raise IOError(f"Saved model {args['resume']} doesn't exist.")

    if args["save_model"] != "never" and args["label"] is None:
        raise ValueError(f"Saving model every {args['save_model']} but no label was specified.")

    utils.init_distributed_mode(args)

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    start_date = utils.get_date()
    results_folder = results_utils.get_save_folder(args["model"], start_date, args["label"])
    log.add_file_headler(results_folder)
    logger.info(f"Command is {' '.join(sys.argv)}")
    logger.info(f'Current configuration:\n{pprint.pformat(args, indent=1, width=100, compact=True)}')

    orders = copy.deepcopy(args["order"])
    del args["order"]
    if orders is not None:
        assert isinstance(orders, list) and len(orders)
        assert all(isinstance(o, list) for o in orders)
        assert all([isinstance(c, int) for o in orders for c in o])
    else:
        orders = [None for _ in range(len(seed_list))]

    avg_inc_accs, last_accs, forgettings = [], [], []
    for i, seed in enumerate(seed_list):
        logger.warning("Launching run {}/{}".format(i + 1, len(seed_list)))
        args["seed"] = seed
        args["device"] = device

        start_time = time.time()

        for avg_inc_acc, last_acc, forgetting in _train(args, start_date, orders[i], i):
            yield avg_inc_acc, last_acc, forgetting, False

        avg_inc_accs.append(avg_inc_acc)
        last_accs.append(last_acc)
        forgettings.append(forgetting)

        logger.info("Training finished in {}s.".format(int(time.time() - start_time)))
        yield avg_inc_acc, last_acc, forgetting, True

    logger.info("Label was: {}".format(args["label"]))

    logger.info(
        "Results done on {} seeds: avg: {}, last: {}, forgetting: {}".format(
            len(seed_list), _aggregate_results(avg_inc_accs), _aggregate_results(last_accs),
            _aggregate_results(forgettings)
        )
    )
    logger.info("Individual results avg: {}".format([round(100 * acc, 2) for acc in avg_inc_accs]))
    logger.info("Individual results last: {}".format([round(100 * acc, 2) for acc in last_accs]))
    logger.info(
        "Individual results forget: {}".format([round(100 * acc, 2) for acc in forgettings])
    )

    logger.info(f"Command was {' '.join(sys.argv)}")


def _train(args, start_date, class_order, run_id):
    _set_global_parameters(args)
    inc_dataset, model = _set_data_model(args, class_order)
    results, results_folder = _set_results(args, start_date)
    model.set_result_folder(results_folder)

    memory, memory_val = None, None
    metric_logger = metrics.MetricLogger(
        inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
    )

    for task_id in range(inc_dataset.n_tasks):
        only_stage0 = args.get('only_stage0', False)
        if only_stage0 and task_id > 0:
            logger.info(f'FLAG only_stage0 is set to [{only_stage0}], and taskid is {task_id} now. Exit.')
            exit(0)
        transform_args = ('input_size', 'color_jitter', 'aa', 'train_interpolation', 'reprob', 'remode', 'recount')
        _args = utils.extract_transform_args(transform_args, args)
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val,
                                                                                repeated_aug=args.get('repeated_aug',
                                                                                                      True),
                                                                                repeat_split=args.get('repeat_split',
                                                                                                      True),
                                                                                trans_args=_args)
        if task_info["task"] == args["max_task"]:
            break

        model.set_task_info(task_info)

        # ---------------
        # 1. Prepare Task
        # ---------------
        model.eval()
        model.before_task(train_loader, val_loader if val_loader else test_loader, inc_dataset)

        # -------------
        # 2. Train Task
        # -------------
        dist.barrier()
        resume_from_chkpt, current_type, resume_from_epoch, = _train_task(args, model, train_loader,
                                                                          val_loader,
                                                                          test_loader, run_id, task_id,
                                                                          task_info)
        # ----------------
        # 3. Conclude Task
        # ----------------
        model.eval()
        dist.barrier()
        torch.cuda.synchronize()

        _after_task(args, model, inc_dataset, run_id, task_id, results_folder, save=not resume_from_chkpt,
                    test_loader=test_loader)

        dist.barrier()
        # ------------
        # 4. Eval Tak
        # ------------
        logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))
        if resume_from_chkpt:
            logger.info(f'Skipping eval on task {task_id}, because of loading from checkpoints.')
            path = os.path.join(args['resume'], "predictions_{}".format(run_id),
                                str(task_id).rjust(len(str(30)), "0") + ".pkl")
            try:
                with open(path, 'rb') as f:
                    ypreds, ytrue = pickle.load(f)
            except Exception as e:
                logger.warning(f'{e}')
                ypreds, ytrue = model.eval_task(test_loader)
        else:
            ypreds, ytrue = model.eval_task(test_loader)
            if args["dump_predictions"] and args["label"] and utils.is_main_process():
                os.makedirs(
                    os.path.join(results_folder, "predictions_{}".format(run_id)), exist_ok=True
                )
                with open(
                        os.path.join(
                            results_folder, "predictions_{}".format(run_id),
                            str(task_id).rjust(len(str(30)), "0") + ".pkl"
                        ), "wb+"
                ) as f:
                    pickle.dump((ypreds, ytrue), f)

        metric_logger.log_task(
            ypreds, ytrue, task_size=task_info["increment"], zeroshot=args.get("all_test_classes")
        )

        if args["label"]:
            logger.info(args["label"])
        logger.info("Avg inc acc: {}.".format(metric_logger.last_results["incremental_accuracy"]))
        logger.info("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
        logger.info(
            "Avg inc acc top5: {}.".format(metric_logger.last_results["incremental_accuracy_top5"])
        )
        logger.info("Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"]))
        logger.info("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
        logger.info("Cord metric: {:.2f}.".format(metric_logger.last_results["cord"]))
        if task_id > 0:
            logger.info(
                "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["old_accuracy"],
                    metric_logger.last_results["avg_old_accuracy"]
                )
            )
            logger.info(
                "New accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["new_accuracy"],
                    metric_logger.last_results["avg_new_accuracy"]
                )
            )
        if args.get("all_test_classes"):
            logger.info(
                "Seen classes: {:.2f}.".format(metric_logger.last_results["seen_classes_accuracy"])
            )
            logger.info(
                "unSeen classes: {:.2f}.".format(
                    metric_logger.last_results["unseen_classes_accuracy"]
                )
            )

        results["results"].append(metric_logger.last_results)

        avg_inc_acc = results["results"][-1]["incremental_accuracy"]
        last_acc = results["results"][-1]["accuracy"]["total"]
        forgetting = results["results"][-1]["forgetting"]
        yield avg_inc_acc, last_acc, forgetting

        memory = model.get_memory()
        memory_val = model.get_val_memory()

    logger.info(
        "Average Incremental Accuracy: {}.".format(results["results"][-1]["incremental_accuracy"])
    )
    if args["label"] is not None:
        results_utils.save_results(
            results, args["label"], args["model"], start_date, run_id, args["seed"]
        )

    del model
    del inc_dataset


# ------------------------
# Lifelong Learning phases
# ------------------------


def _train_task(config, model, train_loader, val_loader, test_loader, run_id, task_id, task_info):
    # pre-check the existence of checkpoints for current task_id
    checkpoints_existence = False
    resume_from_epoch = -1
    current_type = 'baseline_training'
    if config["resume"] is not None:
        path = os.path.join(config["resume"], f"net_{run_id}_task_{task_id}.pth")
        if os.path.exists(path):
            checkpoints_existence = True
    if config["resume"] is not None and checkpoints_existence and not (config["resume_first"] and task_id > 0):
        force_from_stage0 = config.get("force_from_stage0", False)
        print(config['resume'])
        resume_from_epoch, current_type = model.load_parameters(config["resume"], run_id)

        logger.info(
            f'{pprint.pformat({i[0]: i[1].requires_grad for i in model.network.named_parameters()}, sort_dicts=False)}')

        if force_from_stage0:
            logger.info(f'force_from_stage0!')
            model.train_task(train_loader, val_loader if val_loader else test_loader, run_id=run_id, stage0=True,
                             task_id=task_id)
            resume_from_chkpt = False
        else:
            # if False:
            logger.info(f'Resume From Epoch {resume_from_epoch}')
            logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
            model.train_task(train_loader, val_loader if val_loader else test_loader, run_id=run_id,
                             initial_epoch=resume_from_epoch + 1, stage0=False, task_id=task_id,
                             finetune=current_type == 'finetune')
            resume_from_chkpt = False

    elif config["resume"] is not None and os.path.isfile(config["resume"]) and \
            os.path.exists(config["resume"]) and task_id == 0:
        # In case we resume from a single model file, it's assumed to be from the first task.
        resume_from_epoch, current_type = model.load_parameters(config["resume"], run_id)
        # model.network = config["resume"]
        logger.info(
            "Skipping initial training phase {} because reloading pretrained model.".
            format(task_id)
        )
        resume_from_chkpt = True
    else:
        logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
        model.train_task(train_loader, val_loader if val_loader else test_loader)
        resume_from_chkpt = False
    return resume_from_chkpt, current_type, resume_from_epoch


def _after_task(config, model, inc_dataset, run_id, task_id, results_folder, save=True, test_loader=None):
    # pre-check the existence of checkpoints for current task_id
    checkpoints_existence = False
    if config["resume"] is not None:
        path = os.path.join(config["resume"], f"meta_{run_id}_task_{task_id}.pkl")
        if os.path.exists(path):
            checkpoints_existence = True

    if config["resume"] is not None and checkpoints_existence and not (config["resume_first"] and task_id > 0):
        model.load_metadata(config["resume"], run_id)
    else:
        model.after_task_intensive(inc_dataset, test_loader)

    logger.info(f'After task intensive done!!!')
    model.after_task(inc_dataset)

    if save and utils.is_main_process() and config["label"] and (
            config["save_model"] == "task" or
            (config["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1) or
            (config["save_model"] == "first" and task_id == 0)
    ):
        model.save_parameters(results_folder, run_id)
        model.save_metadata(results_folder, run_id)
        # model.save_prompts(results_folder, run_id)


# ----------
# Parameters
# ----------


def _set_results(config, start_date):
    if config["label"]:
        results_folder = results_utils.get_save_folder(config["model"], start_date, config["label"])
    else:
        results_folder = None

    if config["save_model"]:
        logger.info("Model will be save at this rythm: {}.".format(config["save_model"]))

    results = results_utils.get_template_results(config)

    return results, results_folder


def _set_data_model(config, class_order):
    inc_dataset = factory.get_data(config, class_order)
    config["classes_order"] = inc_dataset.class_order

    model = factory.get_model(config)
    model.inc_dataset = inc_dataset

    return inc_dataset, model


def _set_global_parameters(config):
    seed = utils.get_rank() + config["seed"]
    _set_seed(seed, config["threads"], config["no_benchmark"], config["detect_anomaly"])
    # factory.set_device(config)


def _set_seed(seed, nb_threads, no_benchmark, detect_anomaly):
    logger.info("Set seed {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if no_benchmark:
        logger.warning("CUDA algos are not determinists but faster!")
    else:
        logger.warning("CUDA algos are determinists but very slow!")
    torch.backends.cudnn.deterministic = not no_benchmark  # This will slow down training.
    torch.set_num_threads(nb_threads)
    if detect_anomaly:
        logger.info("Will detect autograd anomaly.")
        torch.autograd.set_detect_anomaly(detect_anomaly)


def _set_up_options(args):
    options_paths = args["options"] or []

    autolabel = []
    for option_path in options_paths:
        if not os.path.exists(option_path):
            raise IOError("Not found options file {}.".format(option_path))

        args.update(_parse_options(option_path))

        autolabel.append(os.path.splitext(os.path.basename(option_path))[0])

    return "_".join(autolabel)


def _parse_options(path):
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.load(f, Loader=yaml.FullLoader)
        elif path.endswith(".json"):
            return json.load(f)["config"]
        else:
            raise Exception("Unknown file type {}.".format(path))


# ----
# Misc
# ----


def _aggregate_results(list_results):
    res = str(round(statistics.mean(list_results) * 100, 2))
    if len(list_results) > 1:
        res = res + " +/- " + str(round(statistics.stdev(list_results) * 100, 2))
    return res
