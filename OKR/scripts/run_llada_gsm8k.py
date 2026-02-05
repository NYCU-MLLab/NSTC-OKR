#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from networks.modeling_llada import LLaDAModelLM
from networks.llada_generate import generate as diffusion_generate
from dataloaders.collate_fn_math import extract_answer_gsm8k
from evaluate.parser import extract_answer
from evaluate.grader import math_equal


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def load_dataset_split(local_path: str, split: str, data_dir: str | None = None):
    path = Path(local_path)
    if path.is_dir():
        is_saved = (path / "dataset_dict.json").exists() or (path / "state.json").exists()
        if is_saved:
            ds = load_from_disk(str(path))
            if isinstance(ds, DatasetDict):
                return ds[split]
            return ds
    if data_dir:
        return load_dataset(local_path, split=split, data_dir=data_dir)
    return load_dataset(local_path, split=split)


def _rank_valid_samples(dataset_len: int, rank: int, world_size: int) -> int:
    if dataset_len <= rank:
        return 0
    return (dataset_len - 1 - rank) // world_size + 1


def _truncate_batch(batch: dict, n: int) -> dict:
    if n is None:
        return batch
    if n <= 0:
        return {k: v[:0] if isinstance(v, list) else v for k, v in batch.items()}
    return {k: v[:n] if isinstance(v, list) else v for k, v in batch.items()}


def build_prompts(tokenizer, problems):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [[{"role": "user", "content": prompt}] for prompt in problems]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return problems


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    problems,
    steps,
    gen_length,
    block_length,
    device,
    no_sample,
    temperature,
    cfg_scale,
    remasking,
    logits_eos_inf,
    confidence_eos_eot_inf,
):
    prompts = build_prompts(tokenizer, problems)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    if no_sample:
        temperature = 0.0

    mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    if mask_id is None or mask_id < 0:
        mask_id = 126336

    use_cuda_amp = device.startswith("cuda")
    with torch.autocast(device_type="cuda", enabled=use_cuda_amp, dtype=torch.bfloat16):
        output_ids = diffusion_generate(
            model,
            input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
        )

    start = input_ids.shape[1]
    responses = tokenizer.batch_decode(output_ids[:, start:], skip_special_tokens=True)
    return prompts, responses


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def build_example_id(dataset_name: str, split: str, idx: int) -> str:
    return f"{dataset_name}:{split}:{idx:06d}"


def main():
    parser = argparse.ArgumentParser(
        description="Run LLaDA on GSM8K and write standardized predictions.jsonl"
    )
    parser.add_argument("--ckpt_path", required=True, help="Model checkpoint path")
    parser.add_argument("--local_data_path", default="datasets/gsm8k", help="Dataset path")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--output",
        default="OKR/results/llada_baseline_gsm8k/predictions.jsonl",
        help="Output predictions.jsonl path",
    )
    parser.add_argument(
        "--prompts_output",
        default="",
        help="Optional prompts.jsonl output path",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=8)
    parser.add_argument("--no_sample", default="true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--logits_eos_inf", default="false")
    parser.add_argument("--confidence_eos_eot_inf", default="false")
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--template_id", default="collate_fn_gsm8k")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--keep_shards", default="true")
    parser.add_argument("--metrics_output", default="", help="Optional metrics.json output path")
    args = parser.parse_args()

    no_sample = parse_bool(args.no_sample)
    logits_eos_inf = parse_bool(args.logits_eos_inf)
    confidence_eos_eot_inf = parse_bool(args.confidence_eos_eot_inf)
    keep_shards = parse_bool(args.keep_shards)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
    else:
        rank = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = LLaDAModelLM.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_path,
        torch_dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False).to(device)
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False

    data_dir = "main" if "gsm8k" in args.local_data_path else None
    ds = load_dataset_split(args.local_data_path, split=args.split, data_dir=data_dir)
    ds = ds.with_format("torch")
    if "example_id" not in ds.column_names:
        ds = ds.add_column("example_id", list(range(len(ds))))
    if args.max_examples and args.max_examples > 0:
        ds = ds.select(range(min(len(ds), args.max_examples)))

    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, rank=rank, num_replicas=world_size, shuffle=False)
        valid_left = _rank_valid_samples(len(ds), rank, world_size)
    else:
        valid_left = len(ds)

    def collate_fn_gsm8k_with_id(batch):
        problems = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        example_ids = [item["example_id"] for item in batch]
        return {"problems": problems, "answers": answers, "example_ids": example_ids}

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn_gsm8k_with_id,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False if sampler is None else False,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if distributed:
        shard_path = output_path.parent / f"{output_path.stem}.rank{rank}.jsonl"
    else:
        shard_path = output_path

    prompts_path = Path(args.prompts_output) if args.prompts_output else None
    if prompts_path:
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        if distributed:
            prompts_shard_path = prompts_path.parent / f"{prompts_path.stem}.rank{rank}.jsonl"
        else:
            prompts_shard_path = prompts_path
    else:
        prompts_shard_path = None

    git_commit = get_git_commit()

    pbar = tqdm(dl, disable=rank != 0)
    correct = 0
    total = 0

    with shard_path.open("w", encoding="utf-8") as pred_f:
        if prompts_shard_path:
            prompt_f = prompts_shard_path.open("w", encoding="utf-8")
        else:
            prompt_f = None

        for batch in pbar:
            if valid_left <= 0:
                break
            if len(batch["answers"]) > valid_left:
                batch = _truncate_batch(batch, valid_left)

            prompts, responses = generate_responses(
                model,
                tokenizer,
                batch["problems"],
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                device=device,
                no_sample=no_sample,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
            )

            for i, pred_text in enumerate(responses):
                gold = batch["answers"][i]
                pred_final = extract_answer(pred_text)
                gold_final = extract_answer_gsm8k(gold)
                is_correct = bool(math_equal(gold_final, pred_final))

                example_id = build_example_id("gsm8k", args.split, int(batch["example_ids"][i]))
                prompt = prompts[i]
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

                record = {
                    "example_id": example_id,
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "template_id": args.template_id,
                    "gold": gold,
                    "pred_text": pred_text,
                    "pred_final": pred_final,
                    "gold_final": gold_final,
                    "correct": is_correct,
                    "gen_params": {
                        "steps": args.steps,
                        "gen_length": args.gen_length,
                        "block_length": args.block_length,
                        "temperature": args.temperature,
                        "no_sample": no_sample,
                        "seed": args.seed,
                        "cfg_scale": args.cfg_scale,
                        "remasking": args.remasking,
                        "logits_eos_inf": logits_eos_inf,
                        "confidence_eos_eot_inf": confidence_eos_eot_inf,
                    },
                    "model_name_or_path": args.ckpt_path,
                    "code_git_commit": git_commit,
                }
                pred_f.write(json.dumps(record, ensure_ascii=True) + "\n")

                if prompt_f:
                    prompt_f.write(
                        json.dumps({"example_id": example_id, "prompt": prompt}, ensure_ascii=True)
                        + "\n"
                    )

                total += 1
                if is_correct:
                    correct += 1

            valid_left -= len(batch["answers"])

            if rank == 0:
                acc = (correct / total) if total else 0.0
                pbar.set_description(f"acc: {acc * 100:.2f}%")

        if prompt_f:
            prompt_f.close()

    if distributed:
        totals = torch.tensor([correct, total], device=device, dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        correct, total = int(totals[0].item()), int(totals[1].item())
        dist.barrier()
        if rank == 0:
            with output_path.open("w", encoding="utf-8") as out_f:
                for r in range(world_size):
                    part = output_path.parent / f"{output_path.stem}.rank{r}.jsonl"
                    if part.exists():
                        with part.open("r", encoding="utf-8") as in_f:
                            for line in in_f:
                                out_f.write(line)
            if not keep_shards:
                for r in range(world_size):
                    part = output_path.parent / f"{output_path.stem}.rank{r}.jsonl"
                    if part.exists():
                        part.unlink()

            if prompts_path:
                with prompts_path.open("w", encoding="utf-8") as out_f:
                    for r in range(world_size):
                        part = prompts_path.parent / f"{prompts_path.stem}.rank{r}.jsonl"
                        if part.exists():
                            with part.open("r", encoding="utf-8") as in_f:
                                for line in in_f:
                                    out_f.write(line)
                if not keep_shards:
                    for r in range(world_size):
                        part = prompts_path.parent / f"{prompts_path.stem}.rank{r}.jsonl"
                        if part.exists():
                            part.unlink()

        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        final_acc = (correct / total) if total else 0.0
        print(f"Final Acc: {final_acc * 100:.2f}% ({correct}/{total})", flush=True)
        metrics_path = Path(args.metrics_output) if args.metrics_output else (output_path.parent / "metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": round(final_acc, 4), "num_examples": int(total)},
                f,
                ensure_ascii=True,
                indent=2,
            )


if __name__ == "__main__":
    main()
