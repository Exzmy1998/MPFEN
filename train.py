"""训练 MPFEN：支持主干冻结、混合精度训练和验证集评估。

输入采用 VOC 目录结构，标签像素值为类别索引；背景为 0，裂缝为 1。
权重与训练日志保存到 save_dir，数据目录和模型配置在主入口中设置。
"""

import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.mpfen import MPFEN
from nets.mpfen_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import MPFENDataset, mpfen_dataset_collate
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    # 运行环境：无可用 GPU 时设置 Cuda=False。
    Cuda = True
    seed = 11  # 固定随机种子，便于复现实验。
    # 多卡分布式训练使用 torchrun --nproc_per_node=2 train.py 启动。
    # 当前分布式实现使用 NCCL；常规运行保持 distributed=False。
    distributed = False
    sync_bn = False  # 仅在多卡分布式训练中启用同步 BatchNorm。
    fp16 = False  # 是否启用自动混合精度训练。

    # 模型配置：num_classes 包含背景类，主干可选 mobilenet 或 xception。
    num_classes = 2
    backbone = "mobilenet"
    pretrained = False  # 构建模型时是否加载主干预训练权重。
    # 非空时按参数名及形状加载模型权重；空字符串表示跳过此步骤。
    # 续训需同时设置 Init_Epoch；这里只恢复模型参数，不恢复优化器状态。
    model_path = "logs/best_epoch_weights.pth"
    downsample_factor = 8  # 可选 8 或 16，推理时应与训练配置一致。
    input_shape = [512, 512]  # 输入尺寸，顺序为 [高度, 宽度]。

    # 冻结阶段仅更新主干以外的参数；解冻阶段训练整个网络。
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 32
    UnFreeze_Epoch = 100  # 训练结束轮次，不是解冻阶段的轮次数。
    Unfreeze_batch_size = 16
    Freeze_Train = True  # False 表示从起始轮次直接训练整个网络。

    # 优化器与学习率；实际学习率会根据 batch_size 调整并限制范围。
    Init_lr = 5e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"  # 可选 adam 或 sgd。
    momentum = 0.9  # Adam 的 beta1，或 SGD 的动量。
    weight_decay = 0
    lr_decay_type = 'cos'  # 可选 cos 或 step。

    # 权重保存与验证集评估周期，单位均为 epoch。
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    # 数据集与损失函数配置。
    VOCdevkit_path = 'VOCdevkit'
    dice_loss = True  # 在分类损失基础上叠加 Dice 损失。
    focal_loss = True  # True 使用 Focal 损失，False 使用交叉熵损失。
    cls_weights = np.ones([num_classes], np.float32)  # 每个类别的分类损失权重。
    num_workers = 4  # 数据加载子进程数；0 表示在主进程中加载。

    seed_everything(seed)
    # 设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    # 下载预训练权重
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    model   = MPFEN(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # 加载配置指定的模型权重。
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # 仅载入参数名和张量形状均匹配的权重。
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 输出成功加载及未匹配的参数，便于检查预训练权重。
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # 创建训练损失日志与 TensorBoard 记录。
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    # 使用梯度缩放器支持混合精度训练。
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    # 配置多卡同步 BatchNorm。
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # 将模型包装为分布式数据并行模型。
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # 读取训练集与验证集的样本标识。
    with open(os.path.join(VOCdevkit_path, "Test0717/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "Test0717/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        # 总训练世代指的是遍历全部数据的总次数
        # 总训练步长指的是梯度下降的总次数
        # 每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        # 此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    # 按冻结与解冻阶段配置训练；显存不足时可减小对应 batch_size。
    if True:
        UnFreeze_flag = False
        # 冻结主干参数，保留其余模块的梯度。
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # 如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 根据当前 batch_size 调整学习率。
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # 根据 optimizer_type 创建优化器。
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        # 创建学习率调度函数。
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 计算每轮训练和验证的批次数。
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset   = MPFENDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = MPFENDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = mpfen_dataset_collate, sampler=train_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = mpfen_dataset_collate, sampler=val_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # 记录验证集 mIoU 曲线。
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        # 开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 到达冻结结束轮次后，解冻主干并重新配置数据加载与学习率。
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # 根据当前 batch_size 调整学习率。
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # 创建学习率调度函数。
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = mpfen_dataset_collate, sampler=train_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = mpfen_dataset_collate, sampler=val_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
