"""
Train a diffusion model on images. Phase One 1
"""

import argparse
import json, torch, os
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models, load_tokenizer
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from mytokenizers import SimpleSmilesTokenizer,regexTokenizer
from mydatasets import get_dataloader,ChEBIdataset
import warnings
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")

def main_worker(rank,world_size):
    args = create_argparser().parse_args()
    set_seed(args.seed)

    if rank == 0:
        # 创建tensorboard日志目录
        log_dir = os.path.join(args.checkpoint_path, 'tensorboard_logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        # 记录配置参数
        for key, value in args.__dict__.items():
            writer.add_text('config', f'{key}: {value}')
        
        # 设置Logger的tensorboard writer
        from improved_diffusion.logger import Logger
        Logger.tb_writer = writer
        
        print("Tensorboard logs will be saved to:", log_dir)


    dist_util.setup_dist(rank,world_size) 
    print("creating model and diffusion...")
    smtokenizer = regexTokenizer(max_len=256)
    model = TransformerNetModel2(
        in_channels=args.model_in_channels,  # 3, DEBUG**
        # deep_channels = 10,
        model_channels=args.model_model_channels,
        dropout=args.model_dropout,
        use_checkpoint=False,
        config_name='../../bert-base-uncased',  # 使用本地路径而非huggingface.co的模型名
        training_mode='e2e',
        vocab_size=len(smtokenizer),
        experiment_mode='lm',
        logits_mode=1,
        hidden_size = args.model_hidden_size,
        num_attention_heads=args.model_num_attention_heads,
        num_hidden_layers = args.model_num_hidden_layers,
    )

    if rank==0:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Model total prameter number is :', pytorch_total_params)
        print('Smiles tokenizer vocab length:',len(smtokenizer))

    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(2000)],
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=(
             gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
            )
        ),
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
    )

    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    print('load data', '*'*50)
    
    train_dataset = ChEBIdataset(
        dir='../../datasets/SMILES/',
        smi_tokenizer=smtokenizer,
        split='train_val_256',
        replace_desc=False,
        corrupt_prob=0.,
        mask_desc=False
        # pre = pre
    )
    print('In total',len(train_dataset),'for training....')
    dataloader = get_dataloader(train_dataset,args.batch_size,rank,world_size)
    
    data_valid = None
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        tb_writer=writer if rank == 0 else None,
    ).run_loop()
    
    if rank == 0:
        writer.close()
    
    dist.destroy_process_group()


def create_argparser():
    defaults = dict()
    text_defaults = dict(
        attention_resolutions='16,8',  # 注意力机制的分辨率
        batch_size=64,  # 训练批次大小
        cache_mode='no',  # 缓存模式，'no'表示不使用缓存
        checkpoint_path='../../checkpoints',  # 模型检查点保存路径
        class_cond=False,  # 是否使用类别条件
        commonGen_train='diffusion_lm/common-gen/commongen_data',  # CommonGen数据集路径
        config='ll',  # 配置名称
        config_name='bert-base-uncased',  # 预训练模型配置名称
        data_dir='',  # 数据目录
        dataset_config_name='wikitext-2-raw-v1',  # 数据集配置名称
        dataset_name='wikitext',  # 数据集名称
        diffusion_steps=2000,  # 扩散步数
        dropout=0.1,  # Dropout比率
        e2e_train='',  # 端到端训练数据路径
        ema_rate='0.9999',  # 指数移动平均率
        emb_scale_factor=1.0,  # 嵌入缩放因子
        eval_interval=2000,  # 评估间隔步数
        experiment='random',  # 实验名称
        experiment_mode='lm',  # 实验模式，'lm'表示语言模型
        fp16_scale_growth=0.001,  # FP16缩放增长率
        gradient_clipping=2.4,  # 梯度裁剪阈值
        image_size=8,  # 图像大小（用于兼容图像扩散模型的参数）
        in_channel=16,  # 输入通道数
        learn_sigma=False,  # 是否学习sigma参数
        log_interval=20,  # 日志记录间隔步数
        logits_mode=1,  # logits计算模式
        lr = 0.00005,  # 学习率
        # lr=0.0001, # Lower the learing rate in 2024/5/5 for better convergence 
        lr_anneal_steps=200000,  # 学习率退火总步数，实际的扩散模型训练总步数
        microbatch=-1,  # 微批次大小，-1表示等于batch_size
        modality='e2e-tgt',  # 模态类型
        model_arch='transformer',  # 模型架构，使用transformer
        model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',  # 预训练模型名称或路径，实际上没有使用，使用了本地的 bert-base-uncased。作用：文本编码器，用于处理分子描述文本，将其编码为embeding
        noise_level=0.0,  # 噪声水平
        noise_schedule='sqrt',  # 噪声调度类型，使用平方根调度
        num_channels=128,  # 通道数
        num_heads=4,  # 注意力头数
        num_heads_upsample=-1,  # 上采样时的注意力头数，-1表示与num_heads相同
        num_res_blocks=2,  # 残差块数量
        out_channel=16,  # 输出通道数
        padding_mode='pad',  # 填充模式
        predict_xstart=True,  # 是否预测x_0而非噪声
        preprocessing_num_workers=1,  # 预处理工作线程数
        rescale_learned_sigmas=True,  # 是否重新缩放学习的sigma
        rescale_timesteps=True,  # 是否重新缩放时间步
        resume_checkpoint='',  # 恢复训练的检查点路径，空字符串表示从头开始
        roc_train='diffusion_lm/ROCstory',  # ROCStory数据集路径
        save_interval=10000,  # 保存模型的间隔步数
        schedule_sampler='uniform',  # 时间步采样器类型，使用均匀采样
        seed=19991009,  # 随机种子
        sigma_small=False,  # 是否使用小sigma值
        timestep_respacing='',  # 时间步重采样方式，空字符串表示不重采样
        training_mode='e2e',  # 训练模式，'e2e'表示端到端
        use_bert_tokenizer='no',  # 是否使用BERT分词器
        use_checkpoint=False,  # 是否使用梯度检查点以节省内存
        use_fp16=False,  # 是否使用FP16混合精度训练
        use_kl=False,  # 是否使用KL散度作为损失函数
        use_scale_shift_norm=True,  # 是否使用缩放偏移归一化
        weight_decay=0.0,  # 权重衰减系数
        model_in_channels = 32,  # 模型输入通道数
        model_model_channels = 128,  # 模型中间通道数
        model_dropout = 0.1,  # 模型dropout比率
        model_hidden_size = 1024,  # 模型隐藏层大小
        model_num_attention_heads = 16,  # 模型注意力头数
        model_num_hidden_layers = 12  # 模型隐藏层数量
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICES_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    world_size=1
    mp.spawn(main_worker,args=(world_size,),nprocs=world_size,join=True)