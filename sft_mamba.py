from datetime import datetime
from pytz import timezone
import time
from functools import partial
import wandb
import fire
import tqdm
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy

from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.mamba2 import Mamba2

from model_utils.modeling_llama import LlamaDecoderLayer

from pathlib import Path

from main_utils import (
    load_jsonl_examples,
    get_cosine_lr_decay_fn,
    get_grad_norm,
    save_checkpoint,
    )

from utils import load_bimamba2_ckpts

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import AutoTokenizer

from qat.replace_module import replace_with_learnable_binarylinear

LEARNING_RATE = 2.5e-4
LR_SCHEDULE_TYPE = 'cosine'
END_LEARNING_RATE = 2.5e-5
WARMUP_GRAD_STEPS = 2000
GRAD_NORM_CLIP = 1.
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
ACCELERATOR = 'cuda'
PRECISION = 'bf16-mixed'
RANDOM_SEED = 11111

SFT_DATA_DIR = 'data/tokenized_alpacas.json'
TRAIN_EXAMPLES_PER_CHUNK = 52002
EPOCH = 2

IGNORE_INDEX = -100





def collate_fn(examples, device, tokenizer):
    input_ids = [torch.tensor(item['input_ids'], device=device) for item in examples]
    labels = [torch.tensor(item['labels'], device=device) for item in examples]
    input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )



def train_chunk(fabric,
                tokenizer,
                model,
                optimizer,
                lr_schedule_fn,
                examples,
                per_device_batch_size,
                accumulate_grad_batches,
                epoch_id,
                run_wandb,
                WORKDIR):
    step =len(examples) // (per_device_batch_size * fabric.world_size * accumulate_grad_batches)

    example_batch_idxes = tqdm.trange(
        0, len(examples), per_device_batch_size,
        desc=f'SFT (global_micro_batch_size='
             f'{per_device_batch_size * fabric.world_size}, '
             f'accumulate_grad_batches={accumulate_grad_batches})')
    for i in example_batch_idxes:
        t0 = time.time()

        lr = lr_schedule_fn(step)
        step += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        is_accumulating = (step % accumulate_grad_batches != 0)

        batch = collate_fn(
            examples=examples[i:i+per_device_batch_size], device=fabric.device, tokenizer=tokenizer)
        input_ids, labels, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            labels = labels[..., 1:].contiguous()
            label_mask = mask[..., 1:].contiguous()
            logits = model(input_ids).logits[..., :-1, :].contiguous()

            plant_logits = logits.reshape((-1, logits.size(-1)))
            plant_labels = labels.reshape(-1)
            plant_label_mask = label_mask.reshape(-1)

            sel_logits = plant_logits[plant_label_mask]
            sel_labels = plant_labels[plant_label_mask]

            loss = torch.nn.functional.cross_entropy(
                sel_logits, sel_labels)
                
            fabric.backward(loss / accumulate_grad_batches)

        if not is_accumulating:
            grad_norm = get_grad_norm(model=model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        log = {
            'loss': loss.item(),
            'learning_rate': lr,
            'step': step,
            'speed(#tok/s/gpu)': int(input_ids.numel() / (time.time() - t0)),
        }
        
        if not is_accumulating:
            log['grad_norm'] = grad_norm

        example_batch_idxes.set_postfix(log)
        if run_wandb and fabric.global_rank == 0:
            wandb.log(log)

    save_checkpoint(
        fabric=fabric,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        save_dir=f'{WORKDIR}/ckpt-sft-{epoch_id}')


def main(tag='fully_qat',
         model_size='1.3B',
         model_path = None,
         n_nodes=1,
         n_devices_per_node=8,
         per_device_batch_size=16,
         accumulate_grad_batches=1,
         w_bits = 1,
         use_kd=1,
         run_wandb=False
         ):

    TIMEZONE = timezone('EST')
    DATE = str(datetime.now(tz=TIMEZONE)).split()[0]
    PROJECT_NAME = 'FBI-Mamba'
    WORKDIR = f'fully_qat_record/{tag}_{use_kd}_{model_size}_{w_bits}bit_amber'
    RUN_NAME = f'{WORKDIR}_{DATE}'
    
    Path(WORKDIR).mkdir(exist_ok=True, parents=True)


    fabric = L.Fabric(
        accelerator=ACCELERATOR,
        num_nodes=n_nodes,
        devices=n_devices_per_node,
        precision=PRECISION,
        strategy=FSDPStrategy(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer}),
            activation_checkpointing_policy={Mamba2, GatedMLP, Block, LlamaDecoderLayer},
            cpu_offload=True,
            limit_all_gathers=True,
            )
        )
    fabric.launch()

    if fabric.global_rank == 0:
        if run_wandb:
            wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    fabric.seed_everything(RANDOM_SEED)

    if not Path(f'{model_path}/fabric_ckpt').exists():
        model, tokenizer = load_bimamba2_ckpts(model_size, model_path, True, ["lm_head"], 'column')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
        if model_size == '3B':
            config = MambaConfig(
                d_model = 2560,
                vocab_size=32000, 
                d_intermediate = 0,  
                n_layer = 64,
                ssm_cfg={"layer": "Mamba2"}, 
                attn_cfg = {}, 
                attn_layer_idx = [], 
                pad_vocab_size_multiple=16
                )
        elif model_size == '1.3B':
            config = MambaConfig(
                d_model = 2048,
                d_intermediate = 0,  
                n_layer = 48,
                vocab_size=32000, 
                ssm_cfg={"layer": "Mamba2"}, 
                attn_cfg = {}, 
                attn_layer_idx = [], 
                pad_vocab_size_multiple=16,
                tie_embeddings = True
                )
        elif model_size == '780M':
            config = MambaConfig(
                d_model = 1536,
                d_intermediate = 0,  
                n_layer = 48,
                vocab_size=32000, 
                ssm_cfg={"layer": "Mamba2"}, 
                attn_cfg = {}, 
                attn_layer_idx = [], 
                pad_vocab_size_multiple=16,
                tie_embeddings = True
                )
        elif model_size == '370M':
            config = MambaConfig(
                d_model = 1024,
                d_intermediate = 0,  
                n_layer = 48,
                vocab_size=32000, 
                ssm_cfg={"layer": "Mamba2"}, 
                attn_cfg = {}, 
                attn_layer_idx = [], 
                pad_vocab_size_multiple=16,
                tie_embeddings = True
                )

        print(model_size)
        print(config)
        model = MambaLMHeadModel(config) 

        if fabric.global_rank == 0:
            # print(config)
            total_params = sum(p.numel() for p in model.parameters())
            print('base model', total_params)
                
        model = replace_with_learnable_binarylinear(model, 'column', ['lm_head'])

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(BETA1, BETA2),
            foreach=False)
    
    model, optimizer = fabric.setup(model, optimizer)

    if Path(f'{model_path}/fabric_ckpt').exists():
        fabric.load(
            path=f'{model_path}/fabric_ckpt',
            state={'model': model}
            )

    torch.cuda.empty_cache()

    global_micro_batch_size = per_device_batch_size * fabric.world_size
    total_steps = EPOCH * TRAIN_EXAMPLES_PER_CHUNK // (global_micro_batch_size*accumulate_grad_batches) 
    lr_schedule_fn = get_cosine_lr_decay_fn(
        total_steps=total_steps,
        warmup_steps=total_steps*0.03,
        learning_rate=LEARNING_RATE,
        end_learning_rate=END_LEARNING_RATE)
    
    for epoch_id in range(EPOCH):
        examples = load_jsonl_examples(
            filename=SFT_DATA_DIR,
            n_examples=TRAIN_EXAMPLES_PER_CHUNK,
            shuffle=True,
            global_micro_batch_size=global_micro_batch_size,
            global_rank=fabric.global_rank,
            world_size=fabric.world_size)

        train_chunk(
            fabric=fabric,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            lr_schedule_fn=lr_schedule_fn,
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            run_wandb=run_wandb,
            epoch_id = epoch_id,
            WORKDIR=WORKDIR)


if __name__ == '__main__':
    fire.Fire(main)

# python sft_mamba.py --tag final --model_size 3B  --model_path fully_qat_record/mamba2_3b_1_3B_1bit_amber/ckpt-33