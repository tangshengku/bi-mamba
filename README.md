# Bi-Mamba: Towards Accurate 1-Bit State Space Models [[arXiv]](https://arxiv.org/abs/2411.11843)

&#x20;

## 🚀 Introduction

<p align="center">
    <img src="assets/bimamba.webp" alt="Bi-Mamba" width="300"/>
</p>

Bi-Mamba is a scalable and powerful 1-bit Mamba architecture designed for efficient large language models. Our approach addresses the high computational complexity and memory demands of traditional models while ensuring high performance.

🔥 **Key Features:**

- **1-Bit Quantization**: Reduces weights to a binary setting while maintaining accuracy.
- **Efficient Scaling**: Models available in 780M, 1.3B, and 2.7B sizes.
- **Optimized Training Pipeline**: Uses an autoregressive distillation loss for enhanced learning.
- **Superior Performance**: Outperforms post-training binarization (PTB) and binarization-aware training (BAT) Transformer baselines.

---

## 📊 Benchmarks

### Performance Comparison

#### Mamba-2 780M Model
| Method       | BoolQ | PIQA | HS   | WG   | ARC-e | ARC-c | OBQA | Avg  | Wiki2 | PTB  | C4   |
|-------------|-------|------|------|------|-------|-------|------|------|-------|------|------|
| Mamba-2     | 61.5  | 71.8 | 54.9 | 60.2 | 54.3  | 28.5  | 36.2 | 52.5 | 11.8  | 20.0 | 16.5 |
| GPTQ-3bit   | 44.6  | 62.9 | 40.3 | 53.3 | 40.6  | 26.4  | 30.6 | 42.6 | 152.5 | 192.5 | 186.0 |
| GPTQ-2bit   | 40.4  | 52.3 | 25.7 | 51.3 | 25.6  | 25.1  | 30.2 | 35.2 | 1.6e+8 | 1.3e+8 | 7.3e+7 |
| BiLLM       | 54.1  | 52.9 | 26.9 | 50.6 | 28.5  | 26.5  | 27.2 | 38.1 | 1.8e+4 | 2.4e+4 | 1.5e+4 |
| BitNet-1.58 | 58.2  | 68.1 | 35.1 | 55.2 | 51.8  | 21.4  | 20.0 | 44.3 | -     | -    | -    |
| **Bi-Mamba** | **58.5** | **68.0** | **41.6** | **52.0** | **42.4** | **24.3** | **30.6** | **45.3** | **13.4** | **32.4** | **14.5** |

#### Mamba-2 1.3B Model
| Method       | BoolQ | PIQA | HS   | WG   | ARC-e | ARC-c | OBQA | Avg  | Wiki2 | PTB  | C4   |
|-------------|-------|------|------|------|-------|-------|------|------|-------|------|------|
| Mamba-2     | 64.3  | 73.7 | 59.9 | 61.0 | 60.4  | 33.1  | 37.8 | 55.8 | 10.4  | 17.7 | 14.8 |
| GPTQ-3bit   | 56.8  | 68.2 | 48.5 | 54.4 | 48.0  | 28.8  | 30.4 | 47.8 | 29.3  | 56.5 | 37.3 |
| GPTQ-2bit   | 42.0  | 49.9 | 25.7 | 49.6 | 26.4  | 26.1  | 27.6 | 35.3 | 1.2e+6 | 1.0e+6 | 1.3e+6 |
| BiLLM       | 40.1  | 55.4 | 29.6 | 50.7 | 30.6  | 21.8  | 25.4 | 36.2 | 4943.2 | 3540.8 | 4013.6 |
| BitNet-1.58 | 56.7  | 68.8 | 37.7 | 55.8 | 54.9  | 24.2  | 19.6 | 45.4 | -     | -    | -    |
| **Bi-Mamba** | **60.0** | **68.8** | **47.3** | **55.9** | **48.0** | **26.3** | **32.2** | **48.4** | **11.7** | **29.9** | **12.9** |

#### Mamba-2 2.7B Model
| Method       | BoolQ | PIQA | HS   | WG   | ARC-e | ARC-c | OBQA | Avg  | Wiki2 | PTB  | C4   |
|-------------|-------|------|------|------|-------|-------|------|------|-------|------|------|
| Mamba-2     | 70.7  | 76.3 | 66.6 | 63.9 | 64.8  | 36.3  | 38.8 | 59.6 | 9.1   | 15.3 | 13.3 |
| GPTQ-3bit   | 54.8  | 69.9 | 54.0 | 56.0 | 51.6  | 33.3  | 32.8 | 50.3 | 21.2  | 39.0 | 29.3 |
| GPTQ-2bit   | 45.4  | 49.8 | 25.8 | 52.0 | 25.8  | 25.8  | 26.0 | 35.8 | 2.1e+5 | 2.3e+5 | 1.8e+5 |
| BiLLM       | 52.8  | 53.8 | 27.7 | 53.0 | 29.1  | 25.1  | 28.2 | 38.5 | 8707.0 | 1.7e+4 | 1.3e+4 |
| OneBit      | 63.3  | 67.7 | 52.5 | 58.1 | 41.6  | 29.3  | 34.0 | 49.5 | -     | -    | -    |
| **Bi-Mamba** | **58.0** | **72.5** | **54.3** | **56.1** | **51.4** | **29.1** | **32.6** | **50.6** | **10.0** | **21.9** | **11.3** |


All the best results are highlighted in bold.

---

## 📥 Installation

To set up the environment and install dependencies, run the following commands:

```bash
# Clone the repository
git clone https://github.com/Tangshengku/BiMamba.git
cd Bi-Mamba

# Install required dependencies
pip install -r requirements.txt
```

---

## 🔧 Training
Before training, you should first download the [pre-training dataset](https://huggingface.co/datasets/LLM360/AmberDatasets) and specify the path to ``` train_data_dir ``` in ```sbatch.sh ```
After that, you can run it directly:

```bash
srun python train_bimamba.py --tag mamba2_1.3b --model_size 1.3B --train_data_dir $train_data_dir --use_kd 1 --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 4 --run_wandb
```
or use the sbatch script:
```bash
sbatch sbatch.sh
```
You can find the training script for other model sizes in sbatch ```sbatch.sh ```

---

## 📊 Evaluation

To evaluate the binarized model performance, use:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_bimamba.py --path $bimamba_weight_path --exist_extra_para  --batch_size 16 --model_size 1.3B
```
Also, you can find scripts of other model sizes in ``` eval.sh ```. You can download our pre-trained weight [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/shengkun_tang_mbzuai_ac_ae/ErBLZ1rz8GJMtJcpPMon-BMBd3RqV3aGwvNOB6xacIH7ow?e=x761zD).

---

## GPTQ Mamba2
You can also try GPTQ with our repo. Note: Before using GPTQ to quantize Mamba2 or evaluating the corresponding model after using GPTQ, please modify the code on line 53 of file ```mamba_ssm/modules/mamba2.py``` to: ``` use_mem_eff_path=False ``` 

After that, you simply run: 
```bash 
CUDA_VISIBLE_DEVICES=0 python gptq.py pretrained/mamba2-780m c4 --wbits 3 --true-sequential --act-order --save gptq_mamba/mamba2_780M_3bit_seq/pytorch_model.bin 
```
You can find more scipt of GPTQ in ``` gptq.sh ```



## 🛠 License

This project is released under the **Apache-2.0 license**.

---

## ✏️ Citation

For more details, refer to our paper on [arXiv](https://arxiv.org/abs/2411.11843).

If you find this work useful, please consider citing:

```bibtex
@article{tang2024bi,
  title={Bi-Mamba: Towards Accurate 1-Bit State Space Models},
  author={Tang, Shengkun and Ma, Liqun and Li, Haonan and Sun, Mingjie and Shen, Zhiqiang},
  journal={arXiv preprint arXiv:2411.11843},
  year={2024}
}
```

---

## 🙌 Acknowledgements

This project builds upon open-source frameworks like `FBI-LLM`, `transformers` and `PyTorch`. Special thanks to all contributors! 🎉

