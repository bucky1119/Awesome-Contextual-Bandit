#!/usr/bin/env python3
import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")
DATA_PATH = os.path.join(ROOT, "datasets", "ag_news_tfidf_svd100.npz")

# Keep consistent with run_dataset_step.py aliases.
PRESET_LLM_MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen2_5_7b": os.path.join(ROOT, "models", "huggingface", "Qwen2.5-7B-Instruct"),
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_npz_meta(fp: str) -> Dict[str, Any]:
    d = np.load(fp, allow_pickle=True)
    meta = {}
    if "__meta_json" in d.files:
        try:
            meta = json.loads(str(d["__meta_json"]))
        except Exception:
            meta = {}
    return meta


def _find_latest_result_file(dataset: str, algorithm: str) -> Optional[str]:
    # Include nested results folders and legacy flat folders.
    candidates = glob.glob(os.path.join(RESULTS_DIR, "**", "*.npz"), recursive=True)
    best_fp = None
    best_mtime = -1.0

    for fp in candidates:
        name = os.path.basename(fp)
        if algorithm not in name:
            continue

        try:
            meta = _load_npz_meta(fp)
        except Exception:
            continue

        if meta.get("dataset") != dataset:
            continue
        if meta.get("algorithm") != algorithm:
            continue

        mtime = os.path.getmtime(fp)
        if mtime > best_mtime:
            best_mtime = mtime
            best_fp = fp

    return best_fp


def _resolve_model_path(meta: Dict[str, Any], model_override: Optional[str]) -> str:
    if model_override:
        return PRESET_LLM_MODELS.get(model_override, model_override)

    llm_model = meta.get("llm_model")
    if llm_model:
        return PRESET_LLM_MODELS.get(llm_model, llm_model)

    custom_model = meta.get("custom_model")
    if custom_model:
        return custom_model

    model_name = meta.get("model_name")
    if model_name:
        return PRESET_LLM_MODELS.get(model_name, model_name)

    raise ValueError("Cannot resolve model path. Provide --llm_model explicitly.")


def _topk_hits(scores: np.ndarray, label: int, ks: List[int]) -> Dict[int, int]:
    out = {k: 0 for k in ks}
    order = np.argsort(-scores)
    for k in ks:
        if label in order[:k]:
            out[k] = 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute LLM prior prediction accuracy for ag_news using the same setup "
            "as an at_residual_linucb run."
        )
    )
    parser.add_argument("--result_npz", type=str, default=None, help="Path to at_residual_linucb run npz; default auto-detect latest ag_news run.")
    parser.add_argument("--dataset", type=str, default="ag_news", choices=["ag_news"], help="Dataset name.")
    parser.add_argument("--algorithm", type=str, default="at_residual_linucb", help="Algorithm name in meta.")
    parser.add_argument("--llm_model", type=str, default=None, help="Override model alias/path used for local prior generation.")
    parser.add_argument("--llm_device", type=str, default=None, choices=["auto", "cpu", "cuda"], help="Override device.")
    parser.add_argument("--llm_dtype", type=str, default=None, choices=["auto", "float32", "float16", "bfloat16"], help="Override dtype.")
    parser.add_argument("--llm_cache_dir", type=str, default=None, help="Override prior cache dir.")
    parser.add_argument("--max_rounds", type=int, default=None, help="Only evaluate first N rounds for quick check.")
    parser.add_argument("--save_details", action="store_true", help="Save per-round prediction details as JSONL.")
    args = parser.parse_args()

    result_fp = args.result_npz or _find_latest_result_file(args.dataset, args.algorithm)
    if not result_fp:
        raise FileNotFoundError(
            "No matching result file found. Provide --result_npz or run at_residual_linucb on ag_news first."
        )

    result_npz = np.load(result_fp, allow_pickle=True)
    meta = _load_npz_meta(result_fp)
    if not meta:
        raise ValueError("Result file does not contain __meta_json; cannot reconstruct run config.")

    if meta.get("dataset") != "ag_news":
        raise ValueError(f"This script currently supports ag_news only, got dataset={meta.get('dataset')}")

    prior_backend = meta.get("prior_backend", "unknown")
    if prior_backend != "local_llm":
        raise ValueError(
            f"prior_backend={prior_backend}. LLM prediction accuracy requires local_llm prior backend."
        )

    seed = int(meta.get("seed", 42))
    n_rounds = int(meta.get("n_rounds"))
    cold_start_n = int(meta.get("cold_start_n", 0))
    prompt_style = meta.get("prior_prompt_style", "auto")

    effective_rounds = n_rounds
    if args.max_rounds is not None:
        if args.max_rounds <= 0:
            raise ValueError("--max_rounds must be > 0")
        effective_rounds = min(n_rounds, args.max_rounds)

    model_path = _resolve_model_path(meta, args.llm_model)
    llm_device = args.llm_device if args.llm_device is not None else meta.get("llm_device", "auto")
    llm_dtype = args.llm_dtype if args.llm_dtype is not None else meta.get("llm_dtype", "auto")
    llm_cache_dir = args.llm_cache_dir if args.llm_cache_dir is not None else meta.get(
        "llm_cache_dir", os.path.join(RESULTS_DIR, "prior_cache")
    )

    _set_all_seeds(seed)

    from bandits.data.data_sampler import sample_ag_news_data
    from bandits.llm.local_prior_provider import LocalLLMPriorProvider
    from bandits.llm.prompt_registry import get_action_texts

    total = n_rounds + cold_start_n
    dataset, (opt_r_all, opt_a_all), texts = sample_ag_news_data(
        DATA_PATH,
        total,
        shuffle_rows=True,
        return_texts=True,
    )

    if texts is None:
        raise ValueError("AG News texts are missing in dataset file, cannot evaluate LLM text prior.")

    num_actions = int(np.max(opt_a_all)) + 1
    context_dim = dataset.shape[1] - num_actions

    action_texts = get_action_texts("ag_news", num_actions, prompt_style=prompt_style)

    provider = LocalLLMPriorProvider(
        dataset_name="ag_news",
        model_path=model_path,
        tokenizer_path=meta.get("llm_tokenizer_path", None),
        device=llm_device,
        dtype=llm_dtype,
        cache_dir=llm_cache_dir,
        rebuild_cache=False,
        prompt_style=prompt_style,
        dump_prompts=0,
        debug_dir=os.path.join(RESULTS_DIR, "prior_debug"),
        run_tag="eval_prior_accuracy",
    )

    top1_hits = 0
    top2_hits = 0
    top3_hits = 0

    top1_hits_no_error = 0
    top2_hits_no_error = 0
    top3_hits_no_error = 0
    error_count = 0

    details: List[Dict[str, Any]] = []

    for i in range(effective_rounds):
        row_id = cold_start_n + i
        context = dataset[row_id][:context_dim]
        context_text = str(texts[row_id])
        true_action = int(opt_a_all[row_id])

        scores = provider.get_prior_scores(
            sample_id=row_id,
            context=context,
            action_texts=action_texts,
            context_text=context_text,
        )

        # When prior scoring fails, the provider falls back to uniform scores.
        # Treat uniform output as an "error" case for reporting purposes.
        k = scores.shape[0]
        is_error_prior = k > 1 and np.allclose(scores, 1.0 / float(k))
        if is_error_prior:
            error_count += 1

        pred_top1 = int(np.argmax(scores))
        hits = _topk_hits(scores, true_action, [1, 2, 3])
        top1_hits += hits[1]
        top2_hits += hits[2]
        top3_hits += hits[3]

        if not is_error_prior:
            top1_hits_no_error += hits[1]
            top2_hits_no_error += hits[2]
            top3_hits_no_error += hits[3]

        if args.save_details:
            details.append(
                {
                    "round": i,
                    "sample_id": row_id,
                    "true_action": true_action,
                    "pred_top1": pred_top1,
                    "scores": [float(x) for x in np.asarray(scores).reshape(-1)],
                    "correct_top1": bool(hits[1]),
                    "correct_top2": bool(hits[2]),
                    "correct_top3": bool(hits[3]),
                    "error_prior": bool(is_error_prior),
                }
            )

    denom = float(effective_rounds)
    denom_non_error = float(max(1, effective_rounds - error_count))
    summary = {
        "timestamp": datetime.now().isoformat(),
        "result_file": result_fp,
        "dataset": "ag_news",
        "algorithm": args.algorithm,
        "model_path": model_path,
        "seed": seed,
        "cold_start_n": cold_start_n,
        "n_rounds_from_run": n_rounds,
        "evaluated_rounds": effective_rounds,
        "prior_backend": prior_backend,
        "prompt_style": prompt_style,
        "prior_error_count": int(error_count),
        "prior_error_rate": float(error_count) / denom,
        "top1_accuracy": top1_hits / denom,
        "top2_accuracy": top2_hits / denom,
        "top3_accuracy": top3_hits / denom,
        "top1_accuracy_excluding_errors": top1_hits_no_error / denom_non_error,
        "top2_accuracy_excluding_errors": top2_hits_no_error / denom_non_error,
        "top3_accuracy_excluding_errors": top3_hits_no_error / denom_non_error,
        "top1_correct": int(top1_hits),
        "top2_correct": int(top2_hits),
        "top3_correct": int(top3_hits),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = os.path.dirname(result_fp)
    out_summary_fp = os.path.join(out_dir, "ag_news_llm_prior_accuracy_summary.json")
    with open(out_summary_fp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to: {out_summary_fp}")

    if args.save_details:
        out_details_fp = os.path.join(out_dir, "ag_news_llm_prior_accuracy_details.jsonl")
        with open(out_details_fp, "w", encoding="utf-8") as f:
            for row in details:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved details to: {out_details_fp}")


if __name__ == "__main__":
    main()
