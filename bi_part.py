import argparse
from pathlib import Path
from utils import load_mamba2

def binarize_part(model, binary_part):
    assert binary_part in ['out_proj.weight', 'in_proj.weight', 'conv1d.weight', 'mixer.dt_bias', 'mixer.A_log', 'mixer.D']
    for name, param in model.named_parameters():
        if binary_part in name:
            if binary_part not in ['conv1d.weight']:
                binary_w = param.data.sign()
                mean = param.data.mean(-1, keepdim=True)
                scale = (param.data - mean).abs().mean(-1, keepdim=True)
                param.data = scale * binary_w + mean
            else:
                binary_w = param.data.sign().squeeze(1)
                mean = param.data.mean(-1, keepdim=True).squeeze(1)
                scale = (param.data.squeeze(1) - mean).abs().mean(-1, keepdim=True)
                param.data = (scale * binary_w + mean).unsqueeze(1)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Saved model path",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        help="model size",
    )
    parser.add_argument(
        "--keep_parts",
        type=str,
        default="lm_head"
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
        "--scaling_pattern",
        type=str,
        default="column",
        choices=[
            "row",
            "column",
            "single"
        ],
    )

    for model_dir in [
        'pretrained/mamba2-2.7b',
        'pretrained/mamba2-1.3b',
        'pretrained/mamba2-780m'
    ]:
        for part in [
            'out_proj.weight', 
            'in_proj.weight', 
            'conv1d.weight', 
            'mixer.dt_bias', 
            'mixer.A_log', 
            'mixer.D'
            ]:
            print(model_dir, part)
            model, tokenizer = load_mamba2(model_dir)
            model = binarize_part(model, part)
            save_dir = f'bi_part/{Path(model_dir).stem}-{part}'
            tokenizer.save_pretrained(save_dir)
            model.save_pretrained(save_dir)