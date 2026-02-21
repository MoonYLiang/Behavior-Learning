from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import sys
import io
import torch.nn.functional as F

def _fmt_num(x: float, ndigits: int = 4) -> str:
    s = f"{x:.{ndigits}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s != "" else "0"


def _safe_numpy(t):
    if t is None:
        return None
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _get_bl_unit(block):
    unit = getattr(block, "unit", None)
    if unit is None:
        raise AttributeError("Block has no attribute 'unit' (expected BLBlock.unit).")
    return unit


def _get_lambdas(unit):
    lam = unit.lam 
    if getattr(unit, "constrain_lambda", False):
        eps = float(getattr(unit, "eps", 1e-8))
        lam = F.softplus(lam) + eps
    
    lam_u = _safe_numpy(lam[0])
    lam_c = _safe_numpy(lam[1])
    lam_t = _safe_numpy(lam[2])
    
    return lam_u, lam_c, lam_t


def _get_backbone(model):
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        return backbone
    if hasattr(model, "blocks"):
        return model
    raise AttributeError("Cannot find backbone. Expected model.backbone or model.blocks.")


def _get_blocks(model) -> list:
    backbone = _get_backbone(model)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise AttributeError("Backbone has no attribute 'blocks'.")
    return list(blocks)


def _get_hidden_dims(model) -> Optional[Tuple[int, ...]]:
    backbone = _get_backbone(model)
    v = getattr(backbone, "hidden_dims", None)
    if v is None:
        return None
    return tuple(map(int, v))


def _get_output_linears(model) -> Dict[str, torch.nn.Linear]:
    outs: Dict[str, torch.nn.Linear] = {}

    lin = getattr(model, "linear_out", None)
    if lin is not None and hasattr(lin, "weight"):
        outs["Output Layer (Discrete)"] = lin
        return outs

    head = getattr(model, "head", None)
    if head is None:
        return outs
    
    if isinstance(head, torch.nn.Linear):
        outs["Output Layer"] = head
        return outs

    lin = getattr(head, "linear", None)
    if lin is not None and hasattr(lin, "weight"):
        outs["Output Layer"] = lin

    return outs


def _emit_part_lines(
    part_name: str,
    lam_val: float,
    w_row: np.ndarray,
    b_val: float,
    feature_names: List[str],
    ndigits: int = 4,
    tol: float = 0.0,
) -> List[str]:
    lines = []
    lines.append(f"{part_name}")
    lines.append(f"lambda {_fmt_num(float(lam_val), ndigits)}")

    for name, w in zip(feature_names, w_row):
        w = float(w)
        if abs(w) > tol:
            lines.append(f"----{name}     {_fmt_num(w, ndigits)}")

    if abs(float(b_val)) > tol:
        lines.append(f"----C     {_fmt_num(float(b_val), ndigits)}")

    return lines


def _print_blocks(
    block,
    feature_names: List[str],
    layer_idx: int,
    ndigits: int = 4,
    tol: float = 0.0,
) -> int:

    unit = _get_bl_unit(block)

    num_basis = int(unit.lin_u.out_features)

    lam_u, lam_c, lam_t = _get_lambdas(unit)

    w_u = _safe_numpy(unit.lin_u.weight)
    b_u = _safe_numpy(unit.lin_u.bias)
    w_c = _safe_numpy(unit.lin_c.weight)
    b_c = _safe_numpy(unit.lin_c.bias)
    w_t = _safe_numpy(unit.lin_t.weight)
    b_t = _safe_numpy(unit.lin_t.bias)

    for j in range(num_basis):
        block_id = j + 1
        print(f"--B{layer_idx}{block_id}")

        # U part
        lines = _emit_part_lines(
            "U",
            lam_val=float(lam_u[j]),
            w_row=w_u[j],
            b_val=float(b_u[j]),
            feature_names=feature_names,
            ndigits=ndigits,
            tol=tol,
        )
        for ln in lines:
            print(ln)

        # C part
        lines = _emit_part_lines(
            "C",
            lam_val=float(lam_c[j]),
            w_row=w_c[j],
            b_val=float(b_c[j]),
            feature_names=feature_names,
            ndigits=ndigits,
            tol=tol,
        )
        for ln in lines:
            print(ln)

        # T part
        lines = _emit_part_lines(
            "T",
            lam_val=float(lam_t[j]),
            w_row=w_t[j],
            b_val=float(b_t[j]),
            feature_names=feature_names,
            ndigits=ndigits,
            tol=tol,
        )
        for ln in lines:
            print(ln)

        print("")  

    return num_basis


def _print_core(model, blocks, hidden_dims, feat_names, ndigits, tol, title="BL Model Structure"):
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(f"hidden_dims = {hidden_dims}")
    print(f"feature_dim = {len(feat_names)}")
    print("")

    current_feat_names = feat_names
    
    for layer_idx, block in enumerate(blocks, start=1):
        num_basis = _print_blocks(
            block=block,
            feature_names=current_feat_names,
            layer_idx=layer_idx,
            ndigits=ndigits,
            tol=tol,
        )

        if layer_idx < len(blocks):
            current_feat_names = [f"B{layer_idx}{i+1}" for i in range(num_basis)]

    outs = _get_output_linears(model)
    if len(outs) > 0:
        print("=" * 72)
        print("OUTPUT LINEAR(S)")
        print("=" * 72)
        for name, lin in outs.items():
            w = _safe_numpy(lin.weight)
            b = _safe_numpy(lin.bias) if lin.bias is not None else None
            print(name)
            if getattr(w, "size", 0) and w.size <= 16:
                w_flat = [float(x) for x in w.reshape(-1)]
                w_fmt = ", ".join(_fmt_num(x, ndigits) for x in w_flat)
                print(f"weight values = [{w_fmt}]")

            if b is None:
                print("bias = None")
            else:
                if getattr(b, "size", 0) and b.size <= 16:
                    b_flat = [float(x) for x in b.reshape(-1)]
                    b_fmt = ", ".join(_fmt_num(x, ndigits) for x in b_flat)
                    print(f"bias values = [{b_fmt}]")
            print("")

def export_structure(
    model,
    df=None,
    feature_names=None,
    txt_path: Optional[str] = None,
    ndigits: int = 4,
    tol: float = 0.0,
    title: str = "BL Model Structure",
):

    blocks = _get_blocks(model)
    hidden_dims = _get_hidden_dims(model)

    if feature_names is not None:
        feat_names = list(feature_names)
    elif df is not None:
        feat_names = list(df.columns)
    else:
        unit0 = _get_bl_unit(blocks[0])
        if not hasattr(unit0, "lin_u"):
            raise AttributeError("Cannot infer input dim from BLUnit. Expected 'lin_u'.")
        in_dim = int(unit0.lin_u.in_features)
        feat_names = [f"x{i+1}" for i in range(in_dim)]

    if txt_path is None:
        _print_core(model, blocks, hidden_dims, feat_names, ndigits, tol, title)
        return

    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        _print_core(model, blocks, hidden_dims, feat_names, ndigits, tol, title)
    finally:
        sys.stdout = old

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())