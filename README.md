# MultipleShapeLLM

This repository contains data-preparation, fine-tuning, and evaluation scripts for using Gemma 2 as a surrogate model for multiple metasurface geometries. The scripts work with JSON records containing `instruction`, `input`, and `output` fields and support both forward optical-response prediction and inverse-design evaluation.

This repository provides the data-preparation, fine-tuning, and evaluation code associated with these works.

## Related Publications

The research associated with this project has been published in the following articles:

* [Zhang, Huanshu, et al. "Towards a universal meta-optics solver via large language models." arXiv preprint arXiv:2608.26417 (2026).](https://doi.org/10.48550/arXiv.2608.26417)
* [Zhang, Huanshu, et al. "Towards a universal meta-optics solver via large language models." Nano Letters (2026).](https://doi.org/10.1021/acs.nanolett.6c03323)


## Code overview

- `duplicate_merge_and_shuffle.m` reads the configured training and test JSON files, preserves each top-level JSON object as text, merges the files, reproducibly shuffles the records, and writes combined train/test datasets. The current balancing multiplier is capped at 1, so the generated datasets remain unbalanced. The Windows input paths at the top of the script must be changed for another system.
- `finetune_gemma2_9B_bnb_multiGPU.py` formats the JSON records with the Gemma chat template, determines the maximum sequence length from the processed training data, and fine-tunes `unsloth/gemma-2-9b-it-bnb-4bit` with LoRA. It uses Accelerate/DDP for multi-GPU execution, trains only on assistant responses, stops at the configured epoch, and saves the LoRA adapter, tokenizer, configuration, logs, and MATLAB training statistics.
- `eval_gemma2_9B_bnb_defined_true_vllm.py` continuously watches a configured forward-model run directory for new checkpoints and assigns checkpoints to available GPUs. It performs greedy batched inference with vLLM and the checkpoint's LoRA adapter, robustly parses JSON predictions, calculates per-sample and mean MSE/MAE, and saves logs plus JSON and MATLAB evaluation results.
- `eval_gemma2_9B_bnb_inverse_roundtrip_vllm.py` continuously evaluates inverse-model checkpoints. It records predicted geometry parameters, passes successfully parsed designs through a fixed forward-model checkpoint, compares the reconstructed `T1` and `T2` optical responses with the requested responses, and saves parse counts, round-trip MSE/MAE, JSON records, logs, and MATLAB metrics.

The training and evaluation paths, batch sizes, checkpoint locations, and GPU-memory settings are defined near the top of each Python script and should be reviewed before running them.

## Environment

The following versions were installed in the environment when this README was created:

| Package | Version | Purpose |
| --- | --- | --- |
| Python | 3.13.11 | Python runtime |
| PyTorch | 2.9.0+cu130 | CUDA tensor operations, distributed training, and model execution |
| NumPy | 2.2.6 | Numerical arrays and metric calculations |
| Hugging Face Datasets | 4.3.0 | Loading and preprocessing JSON datasets |
| Unsloth | 2025.12.9 | Efficient 4-bit Gemma loading, LoRA setup, and inference |
| TRL | 0.24.0 | Supervised fine-tuning with `SFTTrainer` |
| Transformers | 4.57.3 | Tokenization, chat templates, callbacks, and model utilities |
| SciPy | 1.16.3 | Writing training and evaluation results to MATLAB `.mat` files |
| Accelerate | 1.12.0 | Multi-GPU process coordination |
| vLLM | 0.13.0 | High-throughput batched evaluation with LoRA adapters |
| tqdm | 4.67.1 | Evaluation progress bars |
| PEFT | 0.18.0 | LoRA parameter-efficient fine-tuning support |
| bitsandbytes | 0.49.0 | 4-bit model quantization support |

The PyTorch build targets CUDA 13.0. NVIDIA CUDA-capable GPUs are required by the current training and evaluation configurations. The MATLAB script uses built-in JSON/file-processing functions and does not require an additional MATLAB toolbox.

## Typical workflow

1. Update the file lists in `duplicate_merge_and_shuffle.m`, then run it in MATLAB to create the merged JSON datasets.
2. Update the dataset and output settings in `finetune_gemma2_9B_bnb_multiGPU.py`, then launch training, for example with `accelerate launch finetune_gemma2_9B_bnb_multiGPU.py`.
3. Update the run/checkpoint paths in the appropriate evaluation script and run it with Python. The evaluator remains active, monitors for new checkpoints, and uses one worker process per available GPU.

## Data availability

The data used in this work can be found at https://drive.google.com/drive/folders/1UR0qnlGxqmtwSQWV53phANqj-Bxr8zj9?usp=drive_link
