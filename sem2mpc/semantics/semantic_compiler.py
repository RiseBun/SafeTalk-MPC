# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import re
import copy
from typing import Any, Dict, Tuple, Optional, Callable

Json = Dict[str, Any]

# ---------------- 基础工具 ----------------

def _extract_jsonish_block(text: str) -> str:
    if not isinstance(text, str) or "{" not in text:
        raise ValueError("No '{' found in LLM output; cannot locate JSON object.")
    s = text[text.find("{"):]
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[:i + 1]
    return s + "}" * max(depth, 0)

def _coerce_to_json_obj(patch_any: Any) -> Json:
    if isinstance(patch_any, dict):
        return patch_any
    if isinstance(patch_any, str):
        block = _extract_jsonish_block(patch_any)
        return json.loads(block)
    raise ValueError("Unsupported patch type, need JSON text or dict")

def _deep_merge(dst: Json, src: Json) -> Json:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

# ---------------- 取/设 & 数值健壮转换 ----------------

def _get(obj: Json, path: str, default=None):
    cur = obj
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

_num_pat = re.compile(r"-?\d+(?:\.\d+)?")

def _as_float(val: Any, fallback: float) -> float:
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            m = _num_pat.search(val)
            if m:
                return float(m.group(0))
            return float(val)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return _as_float(val[0], fallback)
        return float(fallback)
    except Exception:
        return float(fallback)

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _scale(base: Json, path: str, ratio: float, lo: float, hi: float, fallback: float) -> float:
    v = _as_float(_get(base, path, fallback), fallback)
    return _clip(v * float(ratio), lo, hi)

def _push(base: Json, path: str, delta: float, lo: float, hi: float, fallback: float) -> float:
    v = _as_float(_get(base, path, fallback), fallback)
    return _clip(v + float(delta), lo, hi)

def _set(dst: Json, path: str, val: Any):
    cur = dst
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = val

# ---------------- 兜底：相对缩放语义编译 ----------------

def _fast_intensity(s: str) -> str:
    """根据用词判定快的强度：base/strong/max"""
    s = s.lower()
    if any(k in s for k in ["最快", "拉满", "极限", "飞起", "全速"]):
        return "max"
    if any(k in s for k in ["特别快", "超级快", "非常快", "更快", "尽快", "加急", "赶时间", "赶飞机"]):
        return "strong"
    if "快" in s or "急一些" in s:
        return "base"
    return ""

def _heuristic_fallback(instr: str, base_dsl: Json) -> Json:
    s = (instr or "").lower()
    rules = []
    patch: Json = {}

    # 显式安全半径：优先级最高
    m = re.search(r"(?:半径|安全距离|radius)\D*([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        try:
            val = float(m.group(1))
            val = _clip(val, 0.30, 1.20)
            patch = {"obstacle": {"radius": val}}
            patch["_origin"] = "fallback"
            patch["_rules"] = ["radius_numeric"]
            patch["_intensity"] = ""
            return patch
        except Exception:
            pass

    # 语义判定
    inten = _fast_intensity(s)               # "", "base", "strong", "max"
    is_fast = bool(inten)
    is_slow = any(k in s for k in ["慢", "慢点", "更慢", "保守"])
    is_smooth = any(k in s for k in ["更平稳", "更平滑", "更稳", "舒适", "别抖动", "颠簸", "平顺"])
    is_safe = any(k in s for k in ["更安全", "安全一些", "孩子", "小孩", "婴儿", "老人", "孕妇", "病人", "乘客"])

    if is_fast:  rules.append("fast:" + inten)
    if is_slow:  rules.append("slow")
    if is_smooth:rules.append("smooth")
    if is_safe:  rules.append("safe")

    # 读取当前值（带 fallback）
    v_max = _as_float(_get(base_dsl, "constraints.v_max", 0.90), 0.90)
    a_max = _as_float(_get(base_dsl, "constraints.a_max", 1.80), 1.80)
    v_far = _as_float(_get(base_dsl, "speed_cap.v_far", 1.00), 1.00)
    v_near= _as_float(_get(base_dsl, "speed_cap.v_near",0.30), 0.30)
    radius= _as_float(_get(base_dsl, "obstacle.radius", 0.45), 0.45)
    w_track=_as_float(_get(base_dsl, "weights.tracking",1.0), 1.0)
    w_ctrl =_as_float(_get(base_dsl, "weights.control", 1.0), 1.0)
    w_smooth=_as_float(_get(base_dsl, "weights.smooth",  1.0), 1.0)
    u_rate =_as_float(_get(base_dsl, "u_rate_weight",    1.0), 1.0)

    # 上下界
    v_max_lo, v_max_hi = 0.40, 1.50
    a_max_lo, a_max_hi = 0.50, 2.00
    v_far_lo, v_far_hi = 0.60, 1.50
    v_near_lo, v_near_hi = 0.20, 0.60
    w_lo, w_hi = 0.3, 3.0
    u_rate_lo, u_rate_hi = 0.5, 3.0
    rad_lo, rad_hi = 0.30, 1.20

    # “快”的强度倍率
    if inten == "base":
        r_vmax, r_amax, r_vfar, r_vnear, r_track, r_ctrl = 1.12, 1.08, 1.12, 1.07, 1.10, 0.90
    elif inten == "strong":
        r_vmax, r_amax, r_vfar, r_vnear, r_track, r_ctrl = 1.20, 1.15, 1.20, 1.15, 1.20, 0.85
    elif inten == "max":
        r_vmax, r_amax, r_vfar, r_vnear, r_track, r_ctrl = 1.33, 1.25, 1.33, 1.25, 1.30, 0.80
    else:
        r_vmax, r_amax, r_vfar, r_vnear, r_track, r_ctrl = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    if is_fast:
        _set(patch, "constraints.v_max", _scale(base_dsl, "constraints.v_max", r_vmax, v_max_lo, v_max_hi, v_max))
        _set(patch, "constraints.a_max", _scale(base_dsl, "constraints.a_max", r_amax, a_max_lo, a_max_hi, a_max))
        _set(patch, "speed_cap.v_far",   _scale(base_dsl, "speed_cap.v_far",   r_vfar, v_far_lo,  v_far_hi,  v_far))
        _set(patch, "speed_cap.v_near",  _scale(base_dsl, "speed_cap.v_near",  r_vnear,v_near_lo, v_near_hi, v_near))
        _set(patch, "weights.tracking",  _scale(base_dsl, "weights.tracking",  r_track, w_lo, w_hi, w_track))
        _set(patch, "weights.control",   _scale(base_dsl, "weights.control",   r_ctrl,  w_lo, w_hi, w_ctrl))

    if is_slow:
        _set(patch, "constraints.v_max", _scale(base_dsl, "constraints.v_max", 0.70, v_max_lo, v_max_hi, v_max))
        _set(patch, "constraints.a_max", _scale(base_dsl, "constraints.a_max", 0.70, a_max_lo, a_max_hi, a_max))
        _set(patch, "speed_cap.v_far",   _scale(base_dsl, "speed_cap.v_far",   0.80, v_far_lo,  v_far_hi,  v_far))
        _set(patch, "speed_cap.v_near",  _scale(base_dsl, "speed_cap.v_near",  0.80, v_near_lo, v_near_hi, v_near))
        _set(patch, "weights.smooth",    _scale(base_dsl, "weights.smooth",    1.30, w_lo, w_hi, w_smooth))
        _set(patch, "weights.control",   _scale(base_dsl, "weights.control",   1.20, w_lo, w_hi, w_ctrl))
        _set(patch, "u_rate_weight",     _scale(base_dsl, "u_rate_weight",     1.20, u_rate_lo, u_rate_hi, u_rate))

    if is_smooth:
        _set(patch, "constraints.a_max", _scale(base_dsl, "constraints.a_max", 0.80, a_max_lo, a_max_hi, a_max))
        _set(patch, "weights.smooth",    _scale(base_dsl, "weights.smooth",    1.40, w_lo, w_hi, w_smooth))
        _set(patch, "weights.control",   _scale(base_dsl, "weights.control",   1.20, w_lo, w_hi, w_ctrl))
        _set(patch, "u_rate_weight",     _scale(base_dsl, "u_rate_weight",     1.30, u_rate_lo, u_rate_hi, u_rate))

    if is_safe:
        _set(patch, "obstacle.radius",   _scale(base_dsl, "obstacle.radius",   1.33, rad_lo, rad_hi, radius))
        _set(patch, "constraints.a_max", _scale(base_dsl, "constraints.a_max", 0.80, a_max_lo, a_max_hi, a_max))
        _set(patch, "speed_cap.v_far",   _scale(base_dsl, "speed_cap.v_far",   0.85, v_far_lo,  v_far_hi,  v_far))
        _set(patch, "speed_cap.v_near",  _scale(base_dsl, "speed_cap.v_near",  0.90, v_near_lo, v_near_hi, v_near))
        _set(patch, "weights.smooth",    _scale(base_dsl, "weights.smooth",    1.25, w_lo, w_hi, w_smooth))
        _set(patch, "weights.control",   _scale(base_dsl, "weights.control",   1.10, w_lo, w_hi, w_ctrl))

    # 标注来源 & 规则
    patch["_origin"] = "fallback"
    patch["_rules"] = rules
    patch["_intensity"] = inten
    return patch

# ---------------- 主入口 ----------------

def compile_from_text(
    original_task: Json | str,
    instruction: str,
    context: Optional[Dict[str, Any]] = None,
    provider: Optional[Callable[[str, Optional[str], Optional[Dict[str, Any]]], str]] = None
) -> Tuple[Json, Json]:
    base: Json = original_task if isinstance(original_task, dict) else json.loads(original_task)

    dsl_hint = json.dumps(
        {k: base.get(k) for k in ("constraints", "speed_cap", "weights", "u_rate_weight", "obstacle") if k in base},
        ensure_ascii=False
    )

    patch_obj: Json = {}
    llm_raw: Optional[str] = None

    # 1) 先尝试 LLM
    if provider is not None:
        try:
            raw = provider(instruction, dsl_hint, context or {})
            llm_raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            if isinstance(raw, dict):
                patch_obj = raw
            elif isinstance(raw, str) and "{" in raw:
                try:
                    patch_obj = _coerce_to_json_obj(raw)
                except Exception:
                    patch_obj = {}
            if isinstance(patch_obj, dict) and patch_obj:
                patch_obj.setdefault("_origin", "llm")
                if llm_raw is not None:
                    patch_obj.setdefault("_llm_raw", llm_raw[:4000])  # 防止太长
        except Exception:
            patch_obj = {}

    # 2) LLM 无效则兜底
    if not isinstance(patch_obj, dict) or not patch_obj:
        patch_obj = _heuristic_fallback(instruction, base)

    # 3) 合并返回
    patched = _deep_merge(copy.deepcopy(base), patch_obj if isinstance(patch_obj, dict) else {})
    if not isinstance(patch_obj, dict):
        patch_obj = {}
    return patched, patch_obj
