"""
verify_checkpoint.py — Validates pedal_bilstm_checkpoint.pt for corrupt/non-finite tensors.

Checks:
  1. File loads without error
  2. Required keys are present
  3. Every model weight tensor is finite (no NaN / Inf)
  4. Every optimizer state tensor is finite
  5. Checkpoint loads cleanly into BiLSTMPedalRegressor
  6. A forward pass on dummy input produces finite output
"""

import sys
import torch
from bilstm import BiLSTMPedalRegressor

CHECKPOINT_PATH = "pedal_bilstm_checkpoint.pt"
REQUIRED_KEYS = {"epoch", "file_offset", "model_state_dict", "optimizer_state_dict"}

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def check_tensor_dict(label, state_dict):
    """Return list of (key, issue) for every non-finite tensor in state_dict."""
    issues = []
    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        if not torch.isfinite(val).all():
            n_nan = torch.isnan(val).sum().item()
            n_inf = torch.isinf(val).sum().item()
            issues.append((key, f"nan={n_nan}, inf={n_inf}, total={val.numel()}"))
    return issues


def main():
    errors = 0

    # ── 1. Load file ────────────────────────────────────────────────────────
    print(f"\n{INFO} Loading '{CHECKPOINT_PATH}' ...")
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        print(f"{PASS} File loaded successfully.")
    except Exception as e:
        print(f"{FAIL} Could not load checkpoint: {e}")
        sys.exit(1)

    # ── 2. Required keys ────────────────────────────────────────────────────
    missing = REQUIRED_KEYS - set(ckpt.keys())
    if missing:
        print(f"{FAIL} Missing required keys: {missing}")
        errors += 1
    else:
        print(f"{PASS} All required keys present: {sorted(ckpt.keys())}")

    epoch = ckpt.get("epoch", "?")
    file_offset = ckpt.get("file_offset", "?")
    print(f"{INFO} epoch={epoch}  file_offset={file_offset}")

    # ── 3. Model weights ────────────────────────────────────────────────────
    model_sd = ckpt.get("model_state_dict", {})
    print(f"\n{INFO} Checking model_state_dict ({len(model_sd)} tensors) ...")
    model_issues = check_tensor_dict("model", model_sd)
    if model_issues:
        print(f"{FAIL} Non-finite model tensors found:")
        for key, detail in model_issues:
            print(f"       {key}: {detail}")
        errors += len(model_issues)
    else:
        total_params = sum(v.numel() for v in model_sd.values() if isinstance(v, torch.Tensor))
        print(f"{PASS} All model tensors finite. ({total_params:,} parameters)")

    # ── 4. Optimizer state ──────────────────────────────────────────────────
    opt_sd = ckpt.get("optimizer_state_dict", {})
    opt_state = opt_sd.get("state", {})
    flat_opt: dict[str, torch.Tensor] = {}
    for param_id, param_state in opt_state.items():
        for k, v in param_state.items():
            flat_opt[f"param[{param_id}].{k}"] = v

    print(f"\n{INFO} Checking optimizer_state_dict ({len(flat_opt)} state tensors) ...")
    opt_issues = check_tensor_dict("optimizer", flat_opt)
    if opt_issues:
        print(f"{FAIL} Non-finite optimizer tensors found:")
        for key, detail in opt_issues:
            print(f"       {key}: {detail}")
        errors += len(opt_issues)
    else:
        print(f"{PASS} All optimizer tensors finite.")

    # ── 5. Load into model ──────────────────────────────────────────────────
    print(f"\n{INFO} Loading state dict into BiLSTMPedalRegressor ...")
    try:
        model = BiLSTMPedalRegressor()
        missing_keys, unexpected_keys = model.load_state_dict(model_sd, strict=True)
        if missing_keys or unexpected_keys:
            print(f"{FAIL} load_state_dict mismatch — missing: {missing_keys}, unexpected: {unexpected_keys}")
            errors += 1
        else:
            print(f"{PASS} State dict loaded with strict=True, no key mismatches.")
    except Exception as e:
        print(f"{FAIL} load_state_dict raised: {e}")
        errors += 1

    # ── 6. Forward pass ─────────────────────────────────────────────────────
    print(f"\n{INFO} Running forward pass on dummy input (1, 64, 140) ...")
    try:
        model.eval()
        dummy = torch.zeros(1, 64, 140)
        with torch.no_grad():
            out = model(dummy)
        if not torch.isfinite(out).all():
            n_bad = (~torch.isfinite(out)).sum().item()
            print(f"{FAIL} Forward pass produced {n_bad} non-finite output values.")
            errors += 1
        else:
            print(f"{PASS} Forward pass output is finite. shape={tuple(out.shape)}")
    except Exception as e:
        print(f"{FAIL} Forward pass raised: {e}")
        errors += 1

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    if errors == 0:
        print(f"{PASS} Checkpoint is valid. No issues found.")
    else:
        print(f"{FAIL} Checkpoint has {errors} issue(s). See details above.")
    print("=" * 50)
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
