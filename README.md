## Train Bi-Mamba
``` sbatch sbatch.sh ```

## Eval Bi-Mamba
``` sbatch eval.sh ```

## Train Dataset
``` data/AmberDataset/train ```

## GPTQ Mamba2
Before using GPTQ to quantize Mamba2 or evaluating the corresponding model after using GPTQ, please modify the code on line 53 of file ```mamba_ssm/modules/mamba2.py``` to: ``` use_mem_eff_path=False ``` 

base: ``` gptq.py ```

