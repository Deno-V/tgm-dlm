"""
Train a diffusion model on images. Phase Two
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


    dist_util.setup_dist(rank,world_size,port='11323') 
    print("creating model and diffusion...")
    smtokenizer = regexTokenizer(max_len=256)
    model = TransformerNetModel2(
        in_channels=args.model_in_channels,  # 3, DEBUG**
        # deep_channels = 10,
        model_channels=args.model_model_channels,
        dropout=args.model_dropout,
        use_checkpoint=False,
        config_name='bert-base-uncased',
        training_mode='e2e',
        vocab_size=len(smtokenizer),
        experiment_mode='lm',
        logits_mode=1,
        hidden_size = args.model_hidden_size,
        num_attention_heads=args.model_num_attention_heads,
        num_hidden_layers = args.model_num_hidden_layers,
        mask = True
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
        corrupt_prob=0.4,
        mask_desc = False
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
        attention_resolutions='16,8', 
        batch_size=64, 
        cache_mode='no', 
        checkpoint_path='../../correction_checkpoints', 
        class_cond=False, 
        commonGen_train='diffusion_lm/common-gen/commongen_data', 
        config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml', 
        config_name='bert-base-uncased', 
        data_dir='', 
        dataset_config_name='wikitext-2-raw-v1', 
        dataset_name='wikitext', 
        diffusion_steps=2000, 
        dropout=0.1, 
        e2e_train='../datasets/e2e_data', 
        ema_rate='0.9999', 
        emb_scale_factor=1.0, 
        eval_interval=2000, 
        experiment='random', 
        experiment_mode='lm', 
        fp16_scale_growth=0.001, 
        gradient_clipping=2.0, 
        image_size=8, 
        in_channel=16, 
        learn_sigma=False, 
        log_interval=20, 
        logits_mode=1, 
        lr=0.00005, 
        lr_anneal_steps=200000, 
        microbatch=-1, 
        modality='e2e-tgt', 
        model_arch='transformer', 
        model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None', 
        noise_level=0.0, 
        noise_schedule='sqrt', 
        num_channels=128, 
        num_heads=4, 
        num_heads_upsample=-1, 
        num_res_blocks=2, 
        out_channel=16, 
        padding_mode='pad', 
        predict_xstart=True, 
        preprocessing_num_workers=1, 
        rescale_learned_sigmas=True, 
        rescale_timesteps=True, 
        resume_checkpoint='', 
        roc_train='diffusion_lm/ROCstory', 
        save_interval=10000, 
        schedule_sampler='uniform', 
        seed=1334332, 
        sigma_small=False, 
        timestep_respacing='', 
        training_mode='e2e', 
        use_bert_tokenizer='no', 
        use_checkpoint=False, 
        use_fp16=False, 
        use_kl=False, 
        use_scale_shift_norm=True, 
        weight_decay=0.0,
        model_in_channels = 32,
        model_model_channels = 128,
        model_dropout = 0.1,
        model_hidden_size = 1024,
        model_num_attention_heads = 16,
        model_num_hidden_layers = 12
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
