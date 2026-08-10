#!/usr/bin/env python3
"""
Fine-tune Gemma 2 (LoRA, Unsloth) as a physics-surrogate model.
Updated: Spaced-digits with rounding (max 3 decimals), BF16 loading.
Fixed: Dataset processed BEFORE max_seq_length calculation for 100% accuracy.
Remember to update train_on_responses_only when changed model!
"""

import os

# Adjust CUDA device as needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
import copy

# New API: Enable TF32
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, TrainerCallback
from scipy.io import savemat
import sys
from accelerate import Accelerator
import torch.distributed as dist


class StopAtEpochCallback(TrainerCallback):
    """
    Forces the trainer to stop training after reaching a specific epoch,
    allowing the learning rate to be scheduled for a longer duration.
    """
    def __init__(self, stop_epoch: int):
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch is a float, so at the end of epoch 10, it will be 10.0
        if state.epoch >= self.stop_epoch:
            print(f"\n[Callback] Reached epoch {self.stop_epoch}. Halting training early!")
            control.should_training_stop = True

# ─────────────────────────────── Main ──────────────────────────────── #

def main() -> None:
    # ─────────────────────────── Configuration ──────────────────────────── #
    CFG: Dict[str, Any] = dict(
        seed=3407,
        dataset_train_path="data/merged_shuffled_train_8_geometries_unbalanced.json",
        dataset_test_path="data/merged_shuffled_test_8_geometries_unbalanced.json",
        base_model_name="unsloth/gemma-2-9b-it-bnb-4bit",
        lora_rank=64,
        lora_alpha=64,
        lora_dropout=0.0,
        learning_rate=4e-4,
        batch_size=300,
        grad_accum=1,
        weight_decay=1e-2,
        epochs=20,
        stop_epoch=10,  # When the training should stop due to no further accuracy gain and increasing parsing errors
        packing=False,
        max_seq_length_override=None,
        gpu_memory_utilization=1,
        fast_inference=False,
    )

    if CFG['packing']:
        CFG['max_seq_length_override']=4096

    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    # Derived paths

    OUTPUT_ROOT = "results_gemma2_9B_bnb_systematic"
    if is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = None
    # 2. Broadcast the timestamp to all other processes
    # We use a list container because broadcast_object_list expects a list
    timestamp_list = [timestamp]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(timestamp_list, src=0)
    else:
        # single process
        pass
    timestamp = timestamp_list[0]
    output_dir = Path(OUTPUT_ROOT) / f"run_{timestamp}"

    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

        # ───────────────────── Self-copy for reproducibility ─────────────────── #
        try:
            this_file = Path(__file__)
            shutil.copy(this_file, output_dir / this_file.name)
        except NameError:
            pass

        # ───────────────────────────── Logging ───────────────────────────── #
        realtime_log_file = output_dir / "run.log"
        with open(realtime_log_file, "w", encoding="utf-8") as f:
            f.write(f"Real-time Log for run_{timestamp}\n")
            f.write("---------------------------------------------------\n")

        write_to_realtime_log(realtime_log_file, f"Run ID: {timestamp}")
        write_to_realtime_log(realtime_log_file, f"GPU: {gpu_info_string()}")

    # ───────────────────── Environment & Seeds ───────────────────────── #
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG["seed"])

    # ───────────────────────── Load datasets ─────────────────────────── #
    train_path = Path(CFG["dataset_train_path"])
    test_path = Path(CFG["dataset_test_path"])

    # Still loading raw list for logging count, though we use load_dataset below
    train_raw = load_json_list(train_path)
    test_raw = load_json_list(test_path)
    if is_main_process:
        write_to_realtime_log(realtime_log_file, f"Loaded train: {len(train_raw)} items, test: {len(test_raw)} items")

        shutil.copy(train_path, output_dir / train_path.name)
        shutil.copy(test_path, output_dir / test_path.name)

        # ──────────────────────────────────────────────────────────────────────
        # 1. PROCESS DATASET FIRST (Move Up)
        # ──────────────────────────────────────────────────────────────────────
        write_to_realtime_log(realtime_log_file, "Processing dataset and applying spacing logic...")

    # We need a tokenizer to apply the template.
    # We load it temporarily here to format the data and calculate length.
    tok_for_est = AutoTokenizer.from_pretrained(CFG["base_model_name"], use_fast=True)

    train_ds = load_dataset("json", data_files=str(train_path), split="train")

    # Pass spacing logic into the formatter
    fmt_fn = format_batch_builder(tok_for_est)

    to_remove = [c for c in train_ds.column_names if c in ("instruction", "input", "output")]
    with accelerator.main_process_first():
        train_ds = train_ds.map(fmt_fn, batched=True, remove_columns=to_remove, num_proc=64)

    # ──────────────────────────────────────────────────────────────────────
    # 🔎 DEBUG: ONE TRAINING EXAMPLE
    # ──────────────────────────────────────────────────────────────────────
    if is_main_process:
        write_to_realtime_log(realtime_log_file, "\n" + "=" * 80)
        write_to_realtime_log(realtime_log_file, "🔎 DEBUG CHECK: ONE SPACED TRAINING EXAMPLE")
        write_to_realtime_log(realtime_log_file, "-" * 40)
        write_to_realtime_log(realtime_log_file, "--- [FULL CHAT TEMPLATE VIEW] ---")
        write_to_realtime_log(realtime_log_file, train_ds[0]["text"])
        write_to_realtime_log(realtime_log_file, "=" * 80 + "\n")
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # 2. CALCULATE MAX SEQ LENGTH FROM PROCESSED DATA
    # ──────────────────────────────────────────────────────────────────────
    if CFG["max_seq_length_override"] is not None:
        max_seq_length = int(CFG["max_seq_length_override"])
    else:
        if is_main_process:
            write_to_realtime_log(realtime_log_file, "Calculating max_seq_length from processed dataset...")

        # Define a helper to calculate length
        def get_length(example):
            return {"len": len(tok_for_est(example["text"], add_special_tokens=False)["input_ids"])}

        # Run parallel mapping to get lengths
        # We assume dataset_num_proc=64 like previous steps (or use os.cpu_count())
        with accelerator.main_process_first():
            train_ds = train_ds.map(get_length, batched=False, num_proc=64)

        # Extract the list of lengths (fast because it's just accessing the arrow array)
        input_ids_lengths = train_ds["len"]

        if len(input_ids_lengths) > 0:
            max_seq_length = max(input_ids_lengths) + 8  # Add margin
        else:
            raise RuntimeError('Empty input_ids_lengths!')  # Fallback if empty
    if is_main_process:
        write_to_realtime_log(realtime_log_file, f"Max sequence length set to {max_seq_length} (calculated from actual tokenized data)")
    CFG["max_seq_length"] = int(max_seq_length)
    # ─────────────────── Model + LoRA (Unsloth) ─────────────────────── #
    # Load model with the precise max_seq_length calculated above
    model, tokenizer = FastLanguageModel.from_pretrained(
        CFG["base_model_name"],
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
        fast_inference=CFG["fast_inference"],
        max_lora_rank=CFG["lora_rank"],
        gpu_memory_utilization=CFG["gpu_memory_utilization"],
        device_map={"": torch.cuda.current_device()},
    )

    # max_seq_length = min(max_seq_length, int(getattr(tokenizer, "model_max_length", max_seq_length)))
    # print(f'New max_seq_length = {max_seq_length}')
    # CFG["max_seq_length"] = int(max_seq_length)

    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG["lora_rank"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        # use_gradient_checkpointing="unsloth",  # Only enable Unsloth gradient checkpointing if you need the memory
        use_gradient_checkpointing=False,
        random_state=CFG["seed"],
    )

    # ───────────────────────── Trainer config ───────────────────────── #
    sft_args = SFTConfig(
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["learning_rate"],
        group_by_length=False,
        #length_column_name="len",
        packing=CFG['packing'],
        warmup_steps=5,
        logging_steps=1,
        optim="adamw_torch_fused",
        weight_decay=CFG["weight_decay"],
        lr_scheduler_type="linear",
        num_train_epochs=CFG["epochs"],
        save_strategy="epoch",
        output_dir=str(output_dir),
        seed=CFG["seed"],
        report_to="none",
        fp16=not is_bfloat16_supported(),  # <--- Disabled
        bf16=is_bfloat16_supported(),  # <--- Enabled for BF16 training
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
        dataloader_pin_memory=True,
        dataloader_num_workers=min(os.cpu_count(), 32),
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        dataset_num_proc=32,
        ddp_find_unused_parameters=False,  # [ADD] Required for DDP with LoRA
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,  # <--- Passing the pre-processed dataset
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=CFG['packing'],
        args=sft_args,
        callbacks=[StopAtEpochCallback(stop_epoch=CFG['stop_epoch'])]
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # ─────────────────────── GPU / memory stats ─────────────────────── #
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        if is_main_process:
            write_to_realtime_log(realtime_log_file, f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            write_to_realtime_log(realtime_log_file, f"{start_gpu_memory} GB of memory reserved.")

    # ──────────────────────────── Train ─────────────────────────────── #
    if is_main_process:
        write_to_realtime_log(realtime_log_file, "Starting training …")
    trainer_stats = trainer.train()
    if is_main_process:
        write_to_realtime_log(realtime_log_file, "Training done.")

    # ──────────────────────────── Save ──────────────────────────────── #
    lora_dir = output_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    # ──────────────── Logs & Cleanup (Unchanged) ────────────────────── #
    history = [(log["step"], log["loss"]) for log in trainer.state.log_history if "loss" in log]

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    cfg_for_mat = sanitize_config_for_mat(CFG)
    safe_savemat(
        output_dir / "training_stats.mat",
        {
            "config": cfg_for_mat,
            "steps": np.array([s for s, _ in history], dtype=np.int32),
            "losses": np.array([l for _, l in history], dtype=np.float32),
        }
    )

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        if is_main_process:
            write_to_realtime_log(realtime_log_file, f"Peak reserved memory = {used_memory} GB.")
            write_to_realtime_log(realtime_log_file, f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    if is_main_process:
        write_to_realtime_log(realtime_log_file, f"Artifacts saved in {output_dir}")

    accelerator.end_training()


# ───────────────────────────── Utilities ───────────────────────────── #

def gpu_info_string() -> str:
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        return f"{props.name} | {round(props.total_memory / 1024 ** 3, 3)} GB total"
    return "CUDA not available"


# ------------------- PROMPT CONSTRUCTION ------------------- #

def _json_dumps_stable(obj: Any) -> str:
    """Stable JSON serialization."""

    def clean_round(o):
        if isinstance(o, float):
            return round(o, 6)
        elif isinstance(o, dict):
            return {k: clean_round(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [clean_round(e) for e in o]
        return o

    return json.dumps(clean_round(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# 3. CHANGE: Simplified Content Builder
def to_gemma_chat_text(tokenizer, instruction: str, input_field: Any, output_field: Any) -> str:
    """
    Constructs the prompt using the official apply_chat_template.
    The 'messages' list is REQUIRED by HuggingFace to apply special tokens correctly.
    """

    # --- Prepare User Content ---
    no_input = (input_field is None or input_field == "" or (isinstance(input_field, dict) and len(input_field) == 0))

    if no_input:
        user_content = instruction.strip()
    else:
        # Convert input to string if it isn't one
        if isinstance(input_field, str):
            inp_txt = input_field.strip()
        else:
            inp_txt = _json_dumps_stable(input_field)

        user_content = (
            f"{instruction.strip()}\n\n"
            f"Input JSON:\n{inp_txt}\n\n"
            f"Return ONLY a JSON object.\n\n"
        )

    # --- Prepare Model Content ---
    if output_field is None:
        model_content = ""
    elif isinstance(output_field, str):
        model_content = output_field.strip()
    else:
        model_content = _json_dumps_stable(output_field)

    # --- Create Messages List (Mandatory for template) ---
    messages = [
        {"role": "user", "content": user_content},
        {"role": "model", "content": model_content},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_batch_builder(tokenizer):
    def _format_batch(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            # Convert to text
            txt = to_gemma_chat_text(tokenizer, inst, inp, out)
            texts.append(txt)
        return {"text": texts}

    return _format_batch


def sanitize_config_for_mat(cfg: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for k, v in cfg.items():
        if v is None: continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def safe_savemat(path: Path, payload: Dict[str, Any]) -> None:
    try:
        savemat(path, payload)
    except Exception as e:
        print(f"savemat failed: {e}")


def write_to_realtime_log(path_to_log, msg):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path_to_log, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}: {msg}\n")
            print(f"{timestamp}: {msg}\n")
    except:
        pass

if __name__ == "__main__":
    main()