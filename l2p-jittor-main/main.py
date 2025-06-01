# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import jittor as jt
from jittor.optim import LambdaLR
from jittor.optim import SGD, Adam, AdamW

from pathlib import Path
from timm.models import create_model

from datasets import build_continual_dataloader
from engine import *
import models
import utils
from vision_transformer_jittor import VisionTransformer_jittor


import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def create_optimizer_simple(args, model):
    opt_name = args.opt.lower()
    params = model.parameters() 
    # 配置优化器参数
    optim_kwargs = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'eps': getattr(args, 'opt_eps', 1e-8)
    }
    
    # 选择优化器类型
    if opt_name == 'sgd':
        optimizer = SGD(params, momentum=args.momentum, **optim_kwargs)
    elif opt_name == 'adamw':
        optimizer = AdamW(params, betas=getattr(args, 'opt_betas', (0.9, 0.999)), **optim_kwargs)
    else:  # 默认用Adam
        optimizer = Adam(params, betas=getattr(args, 'opt_betas', (0.9, 0.999)), **optim_kwargs)
    
    return optimizer

def create_jittor_scheduler(args, optimizer):
    sched_name = args.sched.lower()
    num_epochs = args.epochs
    
    # 基础学习率设置
    if args.unscale_lr:
        lr = args.lr * args.batch_size * jt.world_size / 512.0
    else:
        lr = args.lr
    # 分调度器类型实现
    if sched_name == 'constant':
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        
    elif sched_name == 'cosine':
        def _cosine_decay(epoch):
            if epoch < args.warmup_epochs:
                return (lr - args.warmup_lr) / args.warmup_epochs * epoch + args.warmup_lr
            progress = (epoch - args.warmup_epochs) / (num_epochs - args.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda=_cosine_decay)
        
    elif sched_name == 'step':
        def _step_decay(epoch):
            decay_rate = args.decay_rate ** (epoch // args.decay_epochs)
            decay_rate = max(decay_rate, args.min_lr / lr)
            return decay_rate
        return LambdaLR(optimizer, lr_lambda=_step_decay)
        
    elif sched_name == 'multistep':
        milestones = [int(num_epochs * x) for x in args.lr_milestones]
        def _multistep_decay(epoch):
            return args.decay_rate ** len([m for m in milestones if epoch >= m])
        return LambdaLR(optimizer, lr_lambda=_multistep_decay)
        
    elif sched_name == 'plateau':
        raise NotImplementedError("Plateau scheduler is not natively supported in Jittor")
        
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")

def main(args):
    utils.init_distributed_mode(args)

    jt.flags.use_cuda = 1
    # 指定使用哪块GPU
    jt.flags.gpu_id = 1

    # fix the seed for reproducibility
    seed = args.seed
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    jt.flags.use_cuda = 1 

    data_loader, class_mask = build_continual_dataloader(args)

    # 从timm模型工厂中加载模型，然后转换为jittor模型
    print(f"Creating original model: {args.model}")
    pytorch_original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    pytorch_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )
    #这里做模型框架的转换,, 
    model = VisionTransformer_jittor()
    original_model = VisionTransformer_jittor()
    pytorch_original_model_state_dict = pytorch_original_model.state_dict()
    pytorch_model_state_dict = pytorch_model.state_dict()
    # 将pytorch模型的参数转换为jittor模型的参数
    for key in pytorch_original_model_state_dict.keys():
        if key in original_model.state_dict():
            original_model.state_dict()[key].copy_(jt.array(pytorch_original_model_state_dict[key].numpy()))
        else:
            print(f"Warning: {key} not found in jittor model state_dict")
    # 将pytorch模型的参数转换为jittor模型的参数
    for key in pytorch_model_state_dict.keys():
        if key in model.state_dict():
            model.state_dict()[key].copy_(jt.array(pytorch_model_state_dict[key].numpy()))
        else:
            print(f"Warning: {key} not found in jittor model state_dict")
     

    if args.freeze:
        # Jittor版 - 冻结original_model的所有参数
        original_model.requires_grad_(False)
        
        # Jittor版 - 冻结指定前缀的参数
        for n, p in model.named_parameters():
            if any(n.startswith(prefix) for prefix in args.freeze):
                p.requires_grad = False

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                # 加载模型参数
                checkpoint = jt.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    
    if args.distributed:
        print("Can't use DistributedDataParallel")
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler = create_jittor_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    # 交叉熵函数
    criterion = jt.nn.CrossEntropyLoss()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)