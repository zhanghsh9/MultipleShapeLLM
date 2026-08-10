#!/usr/bin/env python3
"""
Round-trip Evaluation for Gemma 2 (Unsloth/vLLM).
- Monitors the inverse run folder for checkpoints.
- Records inverse parameter predictions without inverse MSE.
- Runs the parsed parameters through the fixed forward checkpoint.
- Preserves the original evaluator's batching, vLLM, and token-budget settings.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import time
import subprocess
import shutil
import json
import logging
import re
import numpy as np
import torch
import copy
import gc
from unsloth import FastLanguageModel
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from scipy.io import savemat
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# vLLM Imports
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Disable dynamo for evaluation stability
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

# ───────────────────────────── USER CONFIGURATION ───────────────────────────── #
# Inverse run to evaluate, and the fixed forward checkpoint to validate designs.
RUN_DIR_PATH = "results_gemma2_9B_bnb_inverse/run_20260507_160549"
TEST_FILE_NAME = "H_real_imag_inverse_test.json"
FORWARD_CHECKPOINT_PATH = "results_gemma2_9B_bnb_systematic/run_20260409_094756/checkpoint-6093"
FORWARD_INSTRUCTION = "Provide the T1 and T2 of this metasurface."

# 3. DEBUGGING & SPEED (Restored exactly as requested)
BATCH_SIZE = 1024
DEBUG_MODE = True if BATCH_SIZE <= 2 else False

# 4. MEMORY (Restored exactly as requested)
USE_FAST_INFERENCE = True
# Restored logic: 0.9 if fast inference, else 1
GPU_MEMORY_UTILIZATION = 0.9 if USE_FAST_INFERENCE else 1

ROUNDTRIP_JSON_NAME = "inverse_roundtrip_evaluation_text_outputs.json"
INVERSE_JSON_NAME = "inverse_parameter_predictions.json"
ROUNDTRIP_MAT_NAME = "inverse_roundtrip_evaluation_metrics.mat"
REALTIME_LOG_NAME = "inverse_roundtrip_realtime_errors.log"

# ────────────────────────────────────────────────────────────────────────────── #

def main_manager():
    """
    Manager Logic (Parallel Pool):
    1. Detects available GPUs.
    2. Maintains a pool of running subprocesses.
    3. Assigns new checkpoints to free GPUs.
    """
    run_path = Path(RUN_DIR_PATH)
    test_path = run_path / TEST_FILE_NAME
    forward_path = Path(FORWARD_CHECKPOINT_PATH)

    if not forward_path.exists():
        raise RuntimeError(f"Forward checkpoint not found: {forward_path}")

    # 1. Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected via torch.cuda.device_count()!")

    # Stack of available GPU IDs (e.g., [0, 1, 2, 3])
    free_gpus = list(range(num_gpus))
    # Dictionary mapping GPU_ID -> (subprocess_handle, checkpoint_path)
    running_procs = {}

    print(f"Monitoring {run_path} for inverse checkpoints...")
    print(f"Forward checkpoint: {forward_path}")
    print(f"Detected {num_gpus} GPUs available for parallel evaluation.")
    print(f"Mode: {'vLLM/Unsloth (Fast)' if USE_FAST_INFERENCE else 'HuggingFace (Standard)'}")

    # Copy script to run dir for reproducibility
    try:
        if run_path.exists():
            shutil.copy(__file__, run_path / Path(__file__).name)
    except:
        pass

    while True:
        if not run_path.exists():
            print(f"Waiting for run directory...", end="\r")
            time.sleep(10)
            continue

        # --- A. CLEANUP COMPLETED PROCESSES ---
        # Check running processes to see if they finished
        gpus_to_free = []
        for gpu_id, (proc, ckpt_path) in running_procs.items():
            ret_code = proc.poll()  # Returns None if running, exit code if done
            if ret_code is not None:
                if ret_code == 0:
                    print(f"GPU {gpu_id}: Finished {ckpt_path.name}")
                else:
                    print(f"GPU {gpu_id}: Failed {ckpt_path.name} (Code {ret_code})")

                gpus_to_free.append(gpu_id)

        for gpu_id in gpus_to_free:
            del running_procs[gpu_id]
            free_gpus.append(gpu_id)
            # Re-sort to prefer lower GPU IDs for tidiness, optional
            free_gpus.sort()

        # --- B. FIND NEW CANDIDATES ---
        candidates = sorted(list(run_path.glob("checkpoint-*")))
        final_model = run_path / "lora"
        if final_model.exists():
            candidates.append(final_model)

        # Get list of paths currently being processed
        active_paths = [p[1] for p in running_procs.values()]

        to_process = []
        for ckpt in candidates:
            # Skip if done
            if (ckpt / ROUNDTRIP_MAT_NAME).exists():
                continue
            # Skip if currently running
            if ckpt in active_paths:
                continue
            # Check validity
            if (ckpt / "adapter_config.json").exists() or (ckpt / "config.json").exists():
                to_process.append(ckpt)

        # --- C. ASSIGN WORKERS ---
        while free_gpus and to_process:
            gpu_id = free_gpus.pop(0)
            ckpt = to_process.pop(0)  # FIFO Queue

            print(f"Launching {ckpt.name} on GPU {gpu_id}")

            env = os.environ.copy()
            env["TARGET_CHECKPOINT"] = str(ckpt)
            env["TARGET_TESTFILE"] = str(test_path)
            env["TARGET_FORWARD_CHECKPOINT"] = str(forward_path)
            env["TORCHDYNAMO_DISABLE"] = "1"
            # CRITICAL: Assign specific GPU to this subprocess
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Use Popen (Non-blocking) instead of run (Blocking)
            try:
                proc = subprocess.Popen([sys.executable, __file__], env=env)
                running_procs[gpu_id] = (proc, ckpt)
            except Exception as e:
                print(f"Failed to launch worker on GPU {gpu_id}: {e}")
                free_gpus.append(gpu_id)  # Return GPU to pool

        # --- D. STATUS UPDATE & SLEEP ---
        if not running_procs and not to_process:
            print(f"Idle. Waiting for checkpoints... (Checked {len(candidates)})", end="\r")
            time.sleep(10)
        else:
            # Don't sleep too long if things are running, so we catch finishes quickly
            time.sleep(2)


def worker_logic():
    """
    Worker Logic:
    1. Loads ONE model.
    2. Runs inference loop.
    3. Offloads metric calc to ThreadPool (Real-time logging).
    4. Saves final results.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    model_path = Path(os.environ["TARGET_CHECKPOINT"])
    test_path = Path(os.environ["TARGET_TESTFILE"])
    forward_model_path = Path(os.environ.get("TARGET_FORWARD_CHECKPOINT", FORWARD_CHECKPOINT_PATH))

    # NEW: Log file for real-time batch errors
    realtime_log_file = model_path / REALTIME_LOG_NAME
    mat_file = model_path / ROUNDTRIP_MAT_NAME

    # Cleanup logic
    if realtime_log_file.exists() and not mat_file.exists():
        try:
            realtime_log_file.unlink()
        except:
            pass
    with open(realtime_log_file, "w", encoding="utf-8") as f:
        f.write(f"Round-trip Real-time Log for {model_path.name}\n")
        f.write("---------------------------------------------------\n")

    if not test_path.exists():
        raise RuntimeError(f"Test file not found: {test_path}")
    if not forward_model_path.exists():
        raise RuntimeError(f"Forward checkpoint not found: {forward_model_path}")

    # --------------------------- Helpers ---------------------------

    def clean_round(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        elif isinstance(obj, dict):
            return {k: clean_round(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_round(e) for e in obj]
        return obj

    def _json_dumps_stable(obj):
        return json.dumps(clean_round(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def build_user_content(instruction, input_field):
        no_input = (input_field is None or input_field == "" or (
                isinstance(input_field, dict) and len(input_field) == 0))
        inp_txt = input_field.strip() if isinstance(input_field, str) else _json_dumps_stable(input_field)
        return (
            f"{instruction.strip()}\n\n"
            f"Input JSON:\n{inp_txt}\n\n"
            f"Return ONLY a JSON object.\n\n"
        )

    def parse_json_robust(text):
        if not text: return None
        clean = text.strip()
        try:
            return json.loads(clean)
        except:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass
        s = clean.find("{")
        e = clean.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(clean[s:e + 1])
            except:
                pass
        if clean.endswith("}"):
            if not clean.startswith("{") and not clean.startswith('"'):
                try:
                    return json.loads('{"' + clean)
                except:
                    pass
            if clean.startswith('"'):
                try:
                    return json.loads('{' + clean)
                except:
                    pass
        return None

    def calculate_metrics(target, pred):
        if not isinstance(target, dict) or not isinstance(pred, dict): return np.nan, np.nan
        tsq, tabs, cnt = 0.0, 0.0, 0
        for k, tv in target.items():
            if tv is None or k not in pred or pred[k] is None: continue
            try:
                t_arr = np.array(tv, float).flatten()
                p_arr = np.array(pred[k], float).flatten()
                if t_arr.shape == p_arr.shape:
                    diff = p_arr - t_arr
                    tsq += np.sum(diff ** 2)
                    tabs += np.sum(np.abs(diff))
                    cnt += len(t_arr)
            except:
                pass
        if cnt == 0: return np.nan, np.nan
        return tsq / cnt, tabs / cnt

    def background_log_batch(batch_idx, decoded_texts, targets, prompts, log_file_path, is_debug):
        """Runs on CPU thread. Parses inverse parameters and appends parse status."""
        fails = 0
        debug_buffer = []

        if is_debug:
            debug_buffer.append(f"\n{'=' * 40} INVERSE BATCH {batch_idx} {'=' * 40}")

        for j, text in enumerate(decoded_texts):
            tgt = targets[j]
            pred = rt_parse_parameter_prediction(text)

            if pred is None:
                fails += 1

            # DEBUG LOGGING
            if is_debug:
                model_input_view = prompts[j].strip()
                target_view = json.dumps(tgt, separators=(',', ':'))
                raw_output_view = text.strip()
                parsed_view = json.dumps(pred, separators=(',', ':')) if pred else "None"

                debug_buffer.append(f"\n[Ex {j}]")
                debug_buffer.append(f"Model Sees: {model_input_view}")
                debug_buffer.append(f"Expected:   {target_view}")
                debug_buffer.append(f"Raw Output: {raw_output_view}")
                debug_buffer.append(f"Parsed:     {parsed_view}")

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"Inverse batch {batch_idx} | Parse Fails: {fails}/{len(targets)}\n")
            if is_debug and debug_buffer:
                f.write("\n".join(debug_buffer) + "\n")

    def write_to_realtime_log(msg):
        try:
            with open(realtime_log_file, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")
        except:
            pass

    def cleanup_merged_model(path: Path):
        """Safely removes the temporary merged model directory."""
        if path.exists():
            shutil.rmtree(path)
            print(f"🧹 Removed temporary merged model: {path}")

    # --------------------------- Context & Model ---------------------------

    run_root = model_path.parent
    config_json_path = run_root / "config.json"
    determined_max_seq_length = None

    if config_json_path.exists():
        try:
            with open(config_json_path, "r", encoding="utf-8") as f:
                train_cfg = json.load(f)
                val = train_cfg.get("max_seq_length")
                if val:
                    determined_max_seq_length = int(val)
                    write_to_realtime_log(f"📘 Found config.json. Using max_seq_length: {determined_max_seq_length}")
        except:
            raise RuntimeError("⚠️ Could not read config.json.")

    if determined_max_seq_length is None:
        write_to_realtime_log("⚠️ config.json not found. Calculating max_seq_length from test set...")
        try:
            tok_temp = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
            ds_temp = load_dataset("json", data_files={"test": str(test_path)})["test"]

            lengths = []
            for row in ds_temp:
                inst = row.get("instruction", "")
                # final_inp = copy.deepcopy(row.get("input", ""))
                # if isinstance(final_inp, dict) and "parameter" in final_inp:
                    # final_inp["parameter"] = space_digits(final_inp["parameter"])
                # final_out = space_digits(copy.deepcopy(row.get("output", {})))
                final_inp = row.get("input", "")
                final_out = row.get("output", {})
                user_txt = build_user_content(inst, final_inp)
                if isinstance(final_out, str):
                    model_txt = final_out.strip()
                else:
                    model_txt = _json_dumps_stable(final_out)

                msg = [{"role": "user", "content": user_txt}, {"role": "model", "content": model_txt}]
                prompt_str = tok_temp.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                ids = tok_temp(prompt_str, add_special_tokens=False)["input_ids"]
                lengths.append(len(ids))

            if not lengths: raise ValueError("Dataset is empty.")
            determined_max_seq_length = max(lengths) + 16
            write_to_realtime_log(f"📏 Calculated dynamic max_seq_length: {determined_max_seq_length}")
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: Could not determine max_seq_length. Error: {e}")

    # --------------------------- DATASET PREP ---------------------------
    # We load the dataset once here as it is needed for both branches
    write_to_realtime_log("Loading Dataset...")
    ds = load_dataset("json", data_files={"test": str(test_path)})["test"]

    # We need a temporary tokenizer for the prompt building (vLLM will load its own internally)
    # or we can just use the one from model_path to build the prompts.
    # Note: vLLM expects raw strings for chat, but for exact control we often pass the formatted string.
    # To keep your logic EXACT, we will pre-format prompts using the tokenizer.
    tokenizer_for_prompt = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)

    # --------------------------- Max Token Calc ---------------------------
    write_to_realtime_log("Scanning ENTIRE test set to calculate max_new_tokens...")
    possible_new_tokens = []

    # We calculate new tokens needed based on the logic: max_seq - prompt_len
    # Since we have determined_max_seq_length, we use that.
    for i, row in enumerate(ds):
        inst = row.get("instruction", "")
        final_inp = copy.deepcopy(row.get("input", ""))
        # if isinstance(final_inp, dict) and "parameter" in final_inp:
            # final_inp["parameter"] = space_digits(final_inp["parameter"])
        user_txt = build_user_content(inst, final_inp)
        msg = [{"role": "user", "content": user_txt}]
        prompt_str = tokenizer_for_prompt.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        input_len = len(tokenizer_for_prompt(prompt_str).input_ids)
        possible_new_tokens.append(determined_max_seq_length - input_len)

    calculated_max_new_tokens = max(possible_new_tokens) if possible_new_tokens else 2048
    if calculated_max_new_tokens > 4096: calculated_max_new_tokens = 4096
    if calculated_max_new_tokens < 128: calculated_max_new_tokens = 128
    write_to_realtime_log(f"Calculated Max New Tokens: {calculated_max_new_tokens}")

    # --------------------------- Prompt Construction ---------------------------
    prompts = []
    targets_dicts = []
    meta_data = []

    for row in ds:
        inst = row.get("instruction", "")
        raw_inp = row.get("input", "")
        out = row.get("output", {})
        final_inp = copy.deepcopy(raw_inp)
        # if isinstance(final_inp, dict) and "parameter" in final_inp:
            # final_inp["parameter"] = space_digits(final_inp["parameter"])
        user_txt = build_user_content(inst, final_inp)
        msg = [{"role": "user", "content": user_txt}]
        # We pre-format the prompt string
        p_str = tokenizer_for_prompt.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        prompts.append(p_str)
        targets_dicts.append(out)
        meta_data.append({"inst": inst, "inp": raw_inp})

    predictions_text = []
    total = len(prompts)
    executor = ThreadPoolExecutor(max_workers=1)

    # ──────────────────────────────────────────────────────────────────────
    # BRANCHING LOGIC: vLLM (Fast) vs HuggingFace (Standard)
    # ──────────────────────────────────────────────────────────────────────
    # Changed logic to first merge lora to bf16 model rather than directly load it

    '''
    # ──────────────────────────────────────────────────────────────────────
    # BRANCHING LOGIC: vLLM (Fast) vs HuggingFace (Standard)
    # ──────────────────────────────────────────────────────────────────────

    if USE_FAST_INFERENCE:
        # [vLLM PATH - MERGED]
        write_to_realtime_log("⚡ Enabling vLLM Native Engine (Merged 4-bit Model)")

        # 1. Define Temporary Path for the Merged Model
        # We use a timestamp to avoid collisions if you run multiple workers
        temp_merged_path = model_path.parent / f"temp_merged_{model_path.name}_{int(time.time())}"

        try:
            # 2. LOAD & MERGE (using Unsloth)
            write_to_realtime_log(f"Merging QLoRA into Base Model (saving to {temp_merged_path.name})...")

            # Load the adapter using Unsloth (exactly as we would for training/HF inference)
            # We explicitly load in 4-bit because your adapter was trained on top of 4-bit
            merge_model, merge_tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_path),
                max_seq_length=determined_max_seq_length,
                load_in_4bit=True,
                dtype=None,  # Auto-detect (likely BF16 on L40S)
            )

            # Force save as Bfloat16 (Standard "merged_16bit" method in Unsloth)
            # This converts the 4-bit weights + LoRA -> Clean Bfloat16 model
            merge_model.save_pretrained_merged(
                str(temp_merged_path),
                merge_tokenizer,
                save_method="merged_4bit_forced",
            )

            # Free up VRAM from the Unsloth load before starting vLLM
            del merge_model
            del merge_tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            # 3. Initialize vLLM Engine with the MERGED model
            # Note: enable_lora=False because the weights are now baked in!
            llm_engine = LLM(
                model=str(temp_merged_path),
                enable_lora=False,
                max_model_len=determined_max_seq_length,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                # dtype="bfloat16",
                dtype='auto',
                enforce_eager=False
            )

            # 4. Define Sampling Params (Greedy)
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=calculated_max_new_tokens,
                stop_token_ids=[1, 107],
            )

            write_to_realtime_log(f"Starting Inference on {total} examples using vLLM (Merged)...")

            # 5. Inference Loop
            for i in tqdm(range(0, total, BATCH_SIZE), desc=f"Eval {model_path.name}"):
                batch_prompts = prompts[i: i + BATCH_SIZE]
                batch_targets = targets_dicts[i: i + BATCH_SIZE]

                # No lora_request needed anymore
                outputs = llm_engine.generate(
                    batch_prompts,
                    sampling_params,
                    use_tqdm=False
                )

                decoded_batch = [o.outputs[0].text for o in outputs]
                predictions_text.extend(decoded_batch)

                executor.submit(
                    background_log_batch,
                    i,
                    decoded_batch,
                    batch_targets,
                    batch_prompts,
                    realtime_log_file,
                    DEBUG_MODE
                )

        except Exception as e:
            raise RuntimeError(f"❌ Error during Merge/Inference: {e}")
        finally:
            # 6. CRITICAL: Clean up disk space
            # This runs whether inference succeeds or fails
            cleanup_merged_model(temp_merged_path)
    '''

    if USE_FAST_INFERENCE:
        # [vLLM PATH]
        write_to_realtime_log("⚡ Enabling vLLM Native Engine (Official Unsloth Integration)")

        # 1. Get Base Model Name (Required for vLLM to load adapter)
        try:
            with open(model_path / "adapter_config.json", "r") as f:
                adapter_cfg = json.load(f)
                base_model_name = adapter_cfg.get("base_model_name_or_path", "unsloth/gemma-2-9b-it-bnb-4bit")
        except Exception:
            base_model_name = "unsloth/gemma-2-9b-it-bnb-4bit"  # Fallback
            raise RuntimeError("⚠️ Could not read adapter_config.json, using fallback base model.")

        # 2. Initialize vLLM Engine
        # enable_lora=True is key. We set max_lora_rank to 32 (matching your train config) to be safe.
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        eos_id = tok.eos_token_id
        eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
        stop_ids = []
        if eos_id is not None:
            stop_ids.append(eos_id)
        # Only add if it’s a real token (not unk)
        if eot_id is not None and tok.unk_token_id is not None and eot_id != tok.unk_token_id:
            stop_ids.append(eot_id)
        write_to_realtime_log(f"Using base model: {base_model_name}")
        llm_engine = LLM(
            model=base_model_name,
            enable_lora=True,
            max_lora_rank=128,  # Safe upper bound
            max_model_len=determined_max_seq_length,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            dtype="bfloat16",
            enforce_eager=False
        )

        # 3. Define Sampling Params
        # Unsloth/Gemma usually uses greedy decoding for eval
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=calculated_max_new_tokens,
            stop_token_ids=stop_ids,
        )

        # 4. Define LoRA Request
        # This tells vLLM to apply the adapter found at 'model_path'
        lora_req = LoRARequest("gemma_adapter", 1, str(model_path))

        write_to_realtime_log(f"Starting Inference on {total} examples using vLLM...")
        for i in tqdm(range(0, total, BATCH_SIZE), desc=f"Eval {model_path.name}"):
            batch_prompts = prompts[i: i + BATCH_SIZE]
            batch_targets = targets_dicts[i: i + BATCH_SIZE]

            # Generate (vLLM handles batching internally, but we feed it chunks to support your realtime logging)
            # vLLM returns a list of RequestOutput objects
            outputs = llm_engine.generate(
                batch_prompts,
                sampling_params,
                lora_request=lora_req,
                use_tqdm=False
            )

            # Extract text
            decoded_batch = [o.outputs[0].text for o in outputs]
            predictions_text.extend(decoded_batch)

            # Async Log
            executor.submit(
                background_log_batch,
                i,
                decoded_batch,
                batch_targets,
                batch_prompts,
                realtime_log_file,
                DEBUG_MODE
            )


    else:
        # [HUGGINGFACE PATH] - EXACT ORIGINAL LOGIC
        write_to_realtime_log("🐢 Using Standard HuggingFace Inference Mode (FastLanguageModel)")

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_path),
                max_seq_length=determined_max_seq_length,
                load_in_4bit=True,
                fast_inference=True,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                dtype=None
            )
            FastLanguageModel.for_inference(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        write_to_realtime_log(f"Starting Inference on {total} examples...")

        for i in tqdm(range(0, total, BATCH_SIZE), desc=f"Eval {model_path.name}"):
            batch_prompts = prompts[i: i + BATCH_SIZE]
            batch_targets = targets_dicts[i: i + BATCH_SIZE]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=calculated_max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
            decoded_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            predictions_text.extend(decoded_batch)

            executor.submit(
                background_log_batch,
                i,
                decoded_batch,
                batch_targets,
                batch_prompts,
                realtime_log_file,
                DEBUG_MODE
            )

    # --------------------------- Round-trip Final Processing ---------------------------
    executor.shutdown(wait=True)
    write_to_realtime_log("Inverse inference complete. Releasing inverse model before forward evaluation...")

    try:
        del llm_engine
    except Exception:
        pass
    try:
        del lora_req
    except Exception:
        pass
    try:
        del model
        del tokenizer
    except Exception:
        pass
    rt_cleanup_vllm_state()

    rt_finalize_roundtrip(
        inverse_model_path=model_path,
        forward_model_path=forward_model_path,
        predictions_text=predictions_text,
        targets_dicts=targets_dicts,
        meta_data=meta_data,
        realtime_log_file=realtime_log_file,
        mat_file=mat_file,
        write_to_realtime_log=write_to_realtime_log,
    )


def rt_clean_round(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: rt_clean_round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rt_clean_round(e) for e in obj]
    return obj


def rt_json_dumps_stable(obj: Any) -> str:
    return json.dumps(rt_clean_round(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rt_build_user_content(instruction: str, input_field: Any) -> str:
    no_input = (
        input_field is None
        or input_field == ""
        or (isinstance(input_field, dict) and len(input_field) == 0)
    )
    if no_input:
        return instruction.strip()

    inp_txt = input_field.strip() if isinstance(input_field, str) else rt_json_dumps_stable(input_field)
    return (
        f"{instruction.strip()}\n\n"
        f"Input JSON:\n{inp_txt}\n\n"
        f"Return ONLY a JSON object.\n\n"
    )


def rt_parse_json_robust(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    clean = text.strip()
    try:
        obj = json.loads(clean)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    s = clean.find("{")
    e = clean.rfind("}")
    if s != -1 and e > s:
        try:
            obj = json.loads(clean[s:e + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    if clean.endswith("}"):
        if not clean.startswith("{") and not clean.startswith('"'):
            try:
                obj = json.loads('{"' + clean)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
        if clean.startswith('"'):
            try:
                obj = json.loads("{" + clean)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
    return None


def rt_normalize_number(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except Exception:
            return None
        return int(parsed) if parsed.is_integer() else parsed
    return None


def rt_parse_parameter_prediction(text: str) -> Optional[Dict[str, List[Any]]]:
    pred = rt_parse_json_robust(text)
    if not isinstance(pred, dict) or "parameter" not in pred:
        return None

    parameter = pred.get("parameter")
    if not isinstance(parameter, list):
        return None

    normalized = []
    for value in parameter:
        parsed_value = rt_normalize_number(value)
        if parsed_value is None:
            return None
        normalized.append(parsed_value)

    return {"parameter": normalized}


def rt_calculate_metrics(target: Any, pred: Any) -> Tuple[float, float]:
    if not isinstance(target, dict) or not isinstance(pred, dict):
        return np.nan, np.nan
    tsq, tabs, cnt = 0.0, 0.0, 0
    for k, tv in target.items():
        if tv is None or k not in pred or pred[k] is None:
            continue
        try:
            t_arr = np.array(tv, float).flatten()
            p_arr = np.array(pred[k], float).flatten()
            if t_arr.shape == p_arr.shape:
                diff = p_arr - t_arr
                tsq += np.sum(diff ** 2)
                tabs += np.sum(np.abs(diff))
                cnt += len(t_arr)
        except Exception:
            pass
    if cnt == 0:
        return np.nan, np.nan
    return tsq / cnt, tabs / cnt


def rt_build_forward_target(inverse_input: Any) -> Dict[str, Any]:
    if not isinstance(inverse_input, dict):
        return {"T1": None, "T2": None}
    return {
        "T1": copy.deepcopy(inverse_input.get("T1")),
        "T2": copy.deepcopy(inverse_input.get("T2")),
    }


def rt_build_forward_input(inverse_input: Any, parameter_prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(inverse_input, dict):
        return None
    if not isinstance(parameter_prediction, dict) or "parameter" not in parameter_prediction:
        return None
    return {
        "geometry": copy.deepcopy(inverse_input.get("geometry")),
        "wavelength": copy.deepcopy(inverse_input.get("wavelength")),
        "parameter": copy.deepcopy(parameter_prediction["parameter"]),
    }


def rt_determine_max_seq_length(model_path: Path, rows: List[Dict[str, Any]], log_fn) -> int:
    config_json_path = model_path.parent / "config.json"
    determined_max_seq_length = None

    if config_json_path.exists():
        try:
            with open(config_json_path, "r", encoding="utf-8") as f:
                train_cfg = json.load(f)
                val = train_cfg.get("max_seq_length")
                if val:
                    determined_max_seq_length = int(val)
                    log_fn(f"Found config.json for {model_path.name}. Using max_seq_length: {determined_max_seq_length}")
        except Exception:
            raise RuntimeError(f"Could not read config.json at {config_json_path}")

    if determined_max_seq_length is not None:
        return determined_max_seq_length

    log_fn(f"config.json not found for {model_path.name}. Calculating max_seq_length from rows...")
    try:
        tok_temp = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        lengths = []
        for row in rows:
            inst = row.get("instruction", "")
            final_inp = row.get("input", "")
            final_out = row.get("output", {})
            user_txt = rt_build_user_content(inst, final_inp)
            model_txt = final_out.strip() if isinstance(final_out, str) else rt_json_dumps_stable(final_out)
            msg = [{"role": "user", "content": user_txt}, {"role": "model", "content": model_txt}]
            prompt_str = tok_temp.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            ids = tok_temp(prompt_str, add_special_tokens=False)["input_ids"]
            lengths.append(len(ids))

        if not lengths:
            raise ValueError("Dataset is empty.")
        determined_max_seq_length = max(lengths) + 16
        log_fn(f"Calculated dynamic max_seq_length for {model_path.name}: {determined_max_seq_length}")
        return determined_max_seq_length
    except Exception as e:
        raise RuntimeError(f"Could not determine max_seq_length for {model_path}. Error: {e}")


def rt_calculate_max_new_tokens(
    tokenizer,
    rows: List[Dict[str, Any]],
    determined_max_seq_length: int,
    log_fn,
    label: str,
) -> int:
    log_fn(f"Scanning {label} rows to calculate max_new_tokens...")
    possible_new_tokens = []

    for row in rows:
        inst = row.get("instruction", "")
        final_inp = copy.deepcopy(row.get("input", ""))
        user_txt = rt_build_user_content(inst, final_inp)
        msg = [{"role": "user", "content": user_txt}]
        prompt_str = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        input_len = len(tokenizer(prompt_str).input_ids)
        possible_new_tokens.append(determined_max_seq_length - input_len)

    calculated_max_new_tokens = max(possible_new_tokens) if possible_new_tokens else 2048
    if calculated_max_new_tokens > 4096:
        calculated_max_new_tokens = 4096
    if calculated_max_new_tokens < 128:
        calculated_max_new_tokens = 128
    log_fn(f"Calculated {label} max_new_tokens: {calculated_max_new_tokens}")
    return calculated_max_new_tokens


def rt_forward_generate(
    model_path: Path,
    prompts: List[str],
    targets: List[Any],
    max_seq_length: int,
    max_new_tokens: int,
    realtime_log_file: Path,
    log_fn,
) -> List[str]:
    if not prompts:
        return []

    executor = ThreadPoolExecutor(max_workers=1)
    predictions_text = []

    try:
        if USE_FAST_INFERENCE:
            log_fn("Enabling vLLM Native Engine for fixed forward checkpoint")
            try:
                with open(model_path / "adapter_config.json", "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                    base_model_name = adapter_cfg.get("base_model_name_or_path", "unsloth/gemma-2-9b-it-bnb-4bit")
            except Exception:
                raise RuntimeError(f"Could not read adapter_config.json at {model_path}")

            tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            eos_id = tok.eos_token_id
            eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
            stop_ids = []
            if eos_id is not None:
                stop_ids.append(eos_id)
            if eot_id is not None and tok.unk_token_id is not None and eot_id != tok.unk_token_id:
                stop_ids.append(eot_id)

            log_fn(f"Using forward base model: {base_model_name}")
            llm_engine = None
            try:
                llm_engine = LLM(
                    model=base_model_name,
                    enable_lora=True,
                    max_lora_rank=128,
                    max_model_len=max_seq_length,
                    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                    dtype="bfloat16",
                    enforce_eager=False,
                )
                sampling_params = SamplingParams(
                    temperature=0,
                    max_tokens=max_new_tokens,
                    stop_token_ids=stop_ids,
                )
                lora_req = LoRARequest("gemma_forward_adapter", 1, str(model_path))

                for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"Forward {model_path.name}"):
                    batch_prompts = prompts[i: i + BATCH_SIZE]
                    batch_targets = targets[i: i + BATCH_SIZE]
                    outputs = llm_engine.generate(
                        batch_prompts,
                        sampling_params,
                        lora_request=lora_req,
                        use_tqdm=False,
                    )
                    decoded_batch = [o.outputs[0].text for o in outputs]
                    predictions_text.extend(decoded_batch)
                    executor.submit(
                        rt_background_log_forward_batch,
                        i,
                        decoded_batch,
                        batch_targets,
                        batch_prompts,
                        realtime_log_file,
                        DEBUG_MODE,
                    )
            finally:
                try:
                    del llm_engine
                except Exception:
                    pass
                try:
                    del lora_req
                except Exception:
                    pass
                rt_cleanup_vllm_state()
        else:
            log_fn("Using Standard HuggingFace Inference Mode for fixed forward checkpoint")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(model_path),
                    max_seq_length=max_seq_length,
                    load_in_4bit=True,
                    fast_inference=True,
                    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                    dtype=None,
                )
                FastLanguageModel.for_inference(model)
            except Exception as e:
                raise RuntimeError(f"Failed to load forward model: {e}")

            try:
                for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"Forward {model_path.name}"):
                    batch_prompts = prompts[i: i + BATCH_SIZE]
                    batch_targets = targets[i: i + BATCH_SIZE]
                    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
                    decoded_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    predictions_text.extend(decoded_batch)
                    executor.submit(
                        rt_background_log_forward_batch,
                        i,
                        decoded_batch,
                        batch_targets,
                        batch_prompts,
                        realtime_log_file,
                        DEBUG_MODE,
                    )
            finally:
                try:
                    del model
                    del tokenizer
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        executor.shutdown(wait=True)

    return predictions_text


def rt_finalize_roundtrip(
    inverse_model_path: Path,
    forward_model_path: Path,
    predictions_text: List[str],
    targets_dicts: List[Any],
    meta_data: List[Dict[str, Any]],
    realtime_log_file: Path,
    mat_file: Path,
    write_to_realtime_log,
) -> None:
    write_to_realtime_log("Finalizing inverse records and building forward prompts...")

    records = []
    inverse_failed_parse_count = 0

    for k, text in enumerate(predictions_text):
        pred = rt_parse_parameter_prediction(text)
        if pred is None:
            inverse_failed_parse_count += 1

        desired_response = rt_build_forward_target(meta_data[k]["inp"])
        records.append({
            "instruction": meta_data[k]["inst"],
            "input_data": meta_data[k]["inp"],
            "target_dict": targets_dicts[k],
            "prediction_text": text,
            "prediction_dict": pred,
            "inverse_parse_failed": pred is None,
            "forward_instruction": FORWARD_INSTRUCTION,
            "forward_input_data": None,
            "forward_target_dict": desired_response,
            "forward_prediction_text": None,
            "forward_prediction_dict": None,
            "forward_mse": None,
            "forward_mae": None,
            "forward_skipped": pred is None,
            "skip_reason": "inverse_parameter_parse_failed" if pred is None else None,
        })

    inverse_json_records = [
        {
            "instruction": r["instruction"],
            "input_data": r["input_data"],
            "target_dict": r["target_dict"],
            "prediction_text": r["prediction_text"],
            "prediction_dict": r["prediction_dict"],
        }
        for r in records
    ]
    with open(inverse_model_path / INVERSE_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(inverse_json_records, f, indent=2, ensure_ascii=False)

    write_to_realtime_log(
        f"Inverse parameter records saved. Parse failures: {inverse_failed_parse_count}/{len(records)}"
    )

    forward_rows = []
    forward_index_map = []
    for idx, record in enumerate(records):
        pred = record["prediction_dict"]
        if pred is None:
            continue

        forward_input = rt_build_forward_input(record["input_data"], pred)
        if forward_input is None:
            record["forward_skipped"] = True
            record["skip_reason"] = "could_not_build_forward_input"
            continue

        record["forward_input_data"] = forward_input
        forward_rows.append({
            "instruction": FORWARD_INSTRUCTION,
            "input": forward_input,
            "output": record["forward_target_dict"],
        })
        forward_index_map.append(idx)

    if forward_rows:
        forward_max_seq_length = rt_determine_max_seq_length(
            forward_model_path,
            rows=forward_rows,
            log_fn=write_to_realtime_log,
        )
        forward_tokenizer = AutoTokenizer.from_pretrained(str(forward_model_path), use_fast=True)
        forward_max_new_tokens = rt_calculate_max_new_tokens(
            forward_tokenizer,
            rows=forward_rows,
            determined_max_seq_length=forward_max_seq_length,
            log_fn=write_to_realtime_log,
            label="forward",
        )

        forward_prompts = []
        forward_targets = []
        for row in forward_rows:
            user_txt = rt_build_user_content(row["instruction"], copy.deepcopy(row["input"]))
            msg = [{"role": "user", "content": user_txt}]
            prompt_str = forward_tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            forward_prompts.append(prompt_str)
            forward_targets.append(row["output"])

        write_to_realtime_log(
            f"Starting forward optical-response inference on {len(forward_prompts)} parsed inverse designs..."
        )
        forward_predictions_text = rt_forward_generate(
            model_path=forward_model_path,
            prompts=forward_prompts,
            targets=forward_targets,
            max_seq_length=forward_max_seq_length,
            max_new_tokens=forward_max_new_tokens,
            realtime_log_file=realtime_log_file,
            log_fn=write_to_realtime_log,
        )

        for local_idx, text in enumerate(forward_predictions_text):
            record_idx = forward_index_map[local_idx]
            tgt = forward_targets[local_idx]
            pred = rt_parse_json_robust(text)
            mse = np.nan
            mae = np.nan
            if pred is not None:
                mse, mae = rt_calculate_metrics(tgt, pred)

            records[record_idx]["forward_prediction_text"] = text
            records[record_idx]["forward_prediction_dict"] = pred
            records[record_idx]["forward_mse"] = float(mse) if not np.isnan(mse) else None
            records[record_idx]["forward_mae"] = float(mae) if not np.isnan(mae) else None
            records[record_idx]["forward_skipped"] = False
            records[record_idx]["skip_reason"] = None
    else:
        write_to_realtime_log("No parsed inverse designs; skipping forward evaluation.")

    mse_per_row = []
    mae_per_row = []
    forward_failed_parse_count = 0
    forward_skipped_count = 0

    for record in records:
        if record["forward_skipped"]:
            forward_skipped_count += 1
            mse_per_row.append(np.nan)
            mae_per_row.append(np.nan)
            continue

        if record["forward_prediction_dict"] is None:
            forward_failed_parse_count += 1

        mse = record["forward_mse"]
        mae = record["forward_mae"]
        mse_per_row.append(np.nan if mse is None else mse)
        mae_per_row.append(np.nan if mae is None else mae)

    mse_arr = np.array(mse_per_row, dtype=np.float32)
    mae_arr = np.array(mae_per_row, dtype=np.float32)
    global_mse = np.nanmean(mse_arr) if np.any(~np.isnan(mse_arr)) else np.nan
    global_mae = np.nanmean(mae_arr) if np.any(~np.isnan(mae_arr)) else np.nan
    valid_roundtrip_count = int(np.sum(~np.isnan(mse_arr)))

    try:
        savemat(str(mat_file), {
            "MSE_per_row": mse_arr,
            "MAE_per_row": mae_arr,
            "MSE_mean": global_mse,
            "MAE_mean": global_mae,
            "num_failed_inverse_parse": inverse_failed_parse_count,
            "num_failed_forward_parse": forward_failed_parse_count,
            "num_forward_skipped": forward_skipped_count,
            "num_valid_roundtrip": valid_roundtrip_count,
            "num_total": len(records),
        })
    except Exception as e:
        raise RuntimeError(f"Failed saving MAT: {e}")

    try:
        with open(inverse_model_path / ROUNDTRIP_JSON_NAME, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed saving JSON: {e}")

    write_to_realtime_log(
        f"Round-trip evaluation done. Global forward MSE: {global_mse:.6e}; "
        f"inverse parse failures: {inverse_failed_parse_count}; "
        f"forward parse failures: {forward_failed_parse_count}; "
        f"valid round trips: {valid_roundtrip_count}/{len(records)}"
    )


def rt_background_log_forward_batch(
    batch_idx: int,
    decoded_texts: List[str],
    targets: List[Any],
    prompts: List[str],
    log_file_path: Path,
    is_debug: bool,
) -> None:
    batch_mse = []
    fails = 0
    debug_buffer = []

    if is_debug:
        debug_buffer.append(f"\n{'=' * 40} FORWARD BATCH {batch_idx} {'=' * 40}")

    for j, text in enumerate(decoded_texts):
        tgt = targets[j]
        pred = rt_parse_json_robust(text)

        mse = np.nan
        if pred is None:
            fails += 1
        else:
            mse, _ = rt_calculate_metrics(tgt, pred)
            batch_mse.append(mse)

        if is_debug:
            model_input_view = prompts[j].strip()
            target_view = json.dumps(tgt, separators=(",", ":"))
            raw_output_view = text.strip()
            parsed_view = json.dumps(pred, separators=(",", ":")) if pred else "None"
            mse_view = f"{mse:.6e}" if not np.isnan(mse) else "NaN"
            debug_buffer.append(f"\n[Ex {j}]")
            debug_buffer.append(f"Model Sees: {model_input_view}")
            debug_buffer.append(f"Expected:   {target_view}")
            debug_buffer.append(f"Raw Output: {raw_output_view}")
            debug_buffer.append(f"Parsed:     {parsed_view}")
            debug_buffer.append(f"MSE:        {mse_view}")

    valid_mse = [m for m in batch_mse if not np.isnan(m)]
    avg_mse = np.nanmean(valid_mse) if valid_mse else np.nan

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"Forward batch {batch_idx} | MSE: {avg_mse:.6e} | Parse Fails: {fails}/{len(targets)}\n")
        if is_debug and debug_buffer:
            f.write("\n".join(debug_buffer) + "\n")


def rt_cleanup_vllm_state() -> None:
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if "TARGET_CHECKPOINT" in os.environ:
        worker_logic()
    else:
        main_manager()
