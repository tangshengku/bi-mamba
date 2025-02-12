import argparse
import torch
from tqdm import tqdm

from lm_eval.models.huggingface import HFLM
import lm_eval
import torch.nn as nn
from pathlib import Path
import json
from datautils import get_loaders
from transformers import AutoTokenizer
from safetensors import safe_open
from utils import load_json, save_json
from qat.replace_module import replace_with_learnable_binarylinear
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.bimamba_mixer_seq_simple import BiMambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import transformers

@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    # def __init__(self, pretrained="state-spaces/mamba2-2.7b", max_length=2048, batch_size=None, device="cuda",
    #              dtype=torch.float16):
    def __init__(self, mdoel, tokenizer, max_length=1024, batch_size=None, device="cuda",
                 dtype=torch.float16):
        LM.__init__(self)
        # self._model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)
        # self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        # setattr(mdoel, "device", device)
        self._model = mdoel
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self.add_bos_token = False
        self.logits_cache = False
        self.truncation = False

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        for key in ("do_sample", "attention_mask"):
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        # mamba's custom GenerationMixin currently does not support
        # passing stopping criteria.
        # for the time being, we simply generate to max length,
        # then truncate (equivalent result)
        # -- this should be revisited to speed up generation
        # stopping_criteria = stop_sequences_criteria(
        #     self.tokenizer, stop, 1, context.shape[0]
        # )

        return self.model.generate(
            input_ids=context,
            max_new_tokens=max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            unk_token_id=self.tokenizer.unk_token_id,
            # stopping_criteria=stopping_criteria,
            # pad_token_id=self.tokenizer.pad_token_id,
            # use_cache=True,
            **generation_kwargs,
        )

def _parse_eval_task(task_str):
    tasks = [s.strip() for s in task_str.split(',')]
    return tasks


def load_mamba_ckpt(model_name):
    model = MambaLMHeadModel.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return model, tokenizer


def load_bimamba_ckpt(model_size, ckpt_dir, ckpt_type, exist_extra_para):
    assert model_size in ["780M", "1.3B", "2.7B"]
    assert ckpt_type in ['torch', 'hf_st', 'lightning']

    ckpt_dir = Path(ckpt_dir)

    with Path(f'mamba_config/bimamba2_{model_size}.json').open('r') as r_f:
            _config = json.load(r_f)
    config = MambaConfig(**_config)

    print(model_size)
    print(config)
    model = BiMambaLMHeadModel(config).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b', padding_side="left", use_fast=False)

    if exist_extra_para:
        model = replace_with_learnable_binarylinear(model, scaling_pattern="column", keep_parts=["lm_head"])

    weight_dict = {}
    if ckpt_type == 'torch':
        ckpt_plist = [p for p in ckpt_dir.iterdir() if p.suffix == '.bin']
        for p in ckpt_plist:
            _weight_dict = torch.load(p)
            for k,v in _weight_dict.items():
                weight_dict[k] = v

    elif ckpt_type == 'lightning':
        ckpt_plist = [p for p in (ckpt_dir/'fabric_ckpt').iterdir() if p.suffix == '.distcp']
        pass # TODO: add lightning ckpt load

    elif ckpt_type == 'hf_st':
        ckpt_plist = [p for p in ckpt_dir.iterdir() if p.suffix == '.safetensors']
        for p in ckpt_plist:
            with safe_open(p, framework="pt", device="cpu") as f:
                weight_dict.update({key: f.get_tensor(key) for key in f.keys()})

    model.load_state_dict(weight_dict)
    for param in model.parameters():
        param.data = param.data.to(torch.float16)

    return model, tokenizer


@torch.no_grad()
def evaluate_ckpt_task(model, tokenizer, tasks, num_fewshot, batch_size, max_length):
    task_manager = lm_eval.tasks.TaskManager()
    eval_lm = MambaEvalWrapper(model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)
    result = lm_eval.simple_evaluate(eval_lm, tasks = tasks, num_fewshot = num_fewshot, task_manager=task_manager)
    return result


@torch.no_grad()
def evaluate_ckpt_ppl(model, tokenizer, ppl_datasets, max_length, limit = -1):
    results = {}
    for dataset in ppl_datasets:
        _, testloader = get_loaders(dataset, tokenizer)
        testenc = testloader.input_ids
        nsamples = testenc.numel() // max_length
        model.eval()
        nlls = []

        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * max_length) : ((i + 1) * max_length)].to('cuda')
            logits = model(batch).logits
            # hidden_states = outputs[0]  # .to(model.lm_head.weight.device)
            # logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * max_length) : ((i + 1) * max_length)][:, 1:].to('cuda')
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * max_length
            nlls.append(neg_log_likelihood)
            if i == limit:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * max_length))
        print(dataset, ppl.item())
        results[dataset] = ppl.item()

    return results


def eval_bimamba(ckpt_dir, args):
    tasks = _parse_eval_task(args.task)
    print('tasks', tasks)
    eval_ppl = 'ppl' in tasks
    down_stream_tasks = [t for t in tasks if t != 'ppl']

    ckpt_dir = Path(ckpt_dir)
    model, tokenizer = load_bimamba_ckpt(
        model_size = args.model_size,
        ckpt_dir = ckpt_dir,
        ckpt_type = args.ckpt_type,
        exist_extra_para = args.exist_extra_para,
        )

    res = {}
    if eval_ppl:
        ppl_res = evaluate_ckpt_ppl(
            model = model,
            tokenizer = tokenizer,
            ppl_datasets = ['wikitext2', 'ptb', 'c4'],
            max_length = 2048
            )
        res.update(ppl_res)
    if len(down_stream_tasks) > 0:
        task_res = evaluate_ckpt_task(
            model = model,
            tokenizer = tokenizer,
            tasks = down_stream_tasks,
            num_fewshot = 0,
            batch_size = args.batch_size,
            max_length = 2048
            )
        res.update(task_res)

    return res


def evaluate_bimamba_ckpt(args):
    save_dir = Path('eval_result')
    save_dir.mkdir(exist_ok=True, parents=True)

    src_dir = Path(args.path)

    save_p = save_dir / f"{src_dir.name}.json"
    if save_p.exists():
        result = load_json(save_p)
    else:
        result = {}

        ckpt_name = f'ckpt-{src_dir.name}'
        print(ckpt_name)

        res = eval_bimamba(src_dir, args)

        if ckpt_name not in result:
            result[ckpt_name] = res
        else:
            result[ckpt_name].update(res)
    save_json(result, save_p)

def evaluate_mamba_ckpt(args):
    save_dir = Path('eval_result')
    save_dir.mkdir(exist_ok=True, parents=True)
    tasks = _parse_eval_task(args.task)
    print('tasks', tasks)

    eval_ppl = 'ppl' in tasks
    down_stream_tasks = [t for t in tasks if t != 'ppl']


    save_p = save_dir / f"{'_'.join(args.path.split('/'))}.json"
    if save_p.exists():
        result = load_json(save_p)
    else:
        result = {}

    model, tokenizer = load_mamba_ckpt(args.path)

    res = {}
    if eval_ppl:
        ppl_res = evaluate_ckpt_ppl(
            model = model,
            tokenizer = tokenizer,
            ppl_datasets = ['wikitext2', 'ptb', 'c4'],
            max_length = 2048
            )
        res.update(ppl_res)
    if len(down_stream_tasks) > 0:
        task_res = evaluate_ckpt_task(
            model = model,
            tokenizer = tokenizer,
            tasks = down_stream_tasks,
            num_fewshot = 0,
            batch_size = args.batch_size,
            max_length = 2048
            )
        res.update(task_res)

    if '_'.join(args.path.split('/')) not in result:
        result['_'.join(args.path.split('/'))] = res
    else:
        result['_'.join(args.path.split('/'))].update(res)
    save_json(result, save_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--path",
        type=str,
        help="Saved model path",
    )
    parser.add_argument(
        "--eval_open_src",
        action="store_true"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ppl,boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
        help="evaluate tasks",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        help="model size",
    )
    parser.add_argument(
        "--ckpt_type",
        type=str,
        default='torch',
    )
    parser.add_argument(
        "--exist_extra_para",
        action="store_true"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch_size for evaluation"
    )

    args = parser.parse_args()

    if args.eval_open_src:
        evaluate_mamba_ckpt(args)
    else:
        evaluate_bimamba_ckpt(args)