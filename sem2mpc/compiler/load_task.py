# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, List

def _is_path(s: str) -> bool:
    """把 .json 结尾或含路径分隔符的字符串当作路径看待"""
    if not isinstance(s, str):
        return False
    s2 = s.strip().strip('\"').strip("'")
    return s2.lower().endswith('.json') or ('/' in s2) or ('\\' in s2)

def _load_json_from_path(path: str) -> Any:
    # 兼容 UTF-8 带 BOM
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def _try_parse_json_string(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception as e:
        raise TypeError(
            "Patch must be a JSON object. Got a plain string that is not valid JSON.\n"
            f"String snippet: {s[:120]!r}\n"
            "Hint: Ensure your LLM outputs *only* a JSON object, with no extra text."
        ) from e

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def merge(dst: Dict[str, Any], src: Any) -> Dict[str, Any]:
    """合并补丁：dict / JSON 字符串 / 文件路径"""
    if isinstance(src, dict):
        return _deep_merge(dst, src)

    if isinstance(src, str):
        s = src.strip().strip('\"').strip("'")
        if _is_path(s):
            # 即使存在性检测失败也先尝试打开，再报错
            try:
                obj = _load_json_from_path(s)
            except FileNotFoundError:
                abs_try = os.path.abspath(s)
                obj = _load_json_from_path(abs_try)  # 二次尝试
            if not isinstance(obj, dict):
                raise TypeError(f"Patch file {s!r} must contain a JSON object, got {type(obj)}")
            return _deep_merge(dst, obj)
        obj = _try_parse_json_string(s)
        if not isinstance(obj, dict):
            raise TypeError(f"Patch must be a JSON object after parsing string, got {type(obj)}")
        return _deep_merge(dst, obj)

    raise TypeError(f"Patch must be a JSON object or JSON string or file path, got {type(src)}")

def apply_patch(task: Any, patch: Any) -> Dict[str, Any]:
    """兼容旧接口：task 可为 dict/路径/JSON 字符串；patch 同上。"""
    base = load_task(task)
    base_copy = json.loads(json.dumps(base))
    return merge(base_copy, patch)

def apply_patches(task: Any, patches: List[Any]) -> Dict[str, Any]:
    cur = load_task(task)
    cur = json.loads(json.dumps(cur))
    for p in patches:
        cur = merge(cur, p)
    return cur

def load_task(task_or_path: Any) -> Dict[str, Any]:
    """读任务 JSON 或对象；支持可选 'base' 合并"""
    if isinstance(task_or_path, dict):
        user = task_or_path
    elif isinstance(task_or_path, str):
        s = task_or_path.strip().strip('\"').strip("'")
        if _is_path(s):
            # 优先当文件打开（相对/绝对都尝试）
            try:
                user = _load_json_from_path(s)
            except FileNotFoundError:
                abs_try = os.path.abspath(s)
                user = _load_json_from_path(abs_try)
        else:
            user = _try_parse_json_string(s)
    else:
        raise TypeError(f"Task must be a dict, JSON string, or path, got {type(task_or_path)}")

    if not isinstance(user, dict):
        raise TypeError(f"Task must be a JSON object, got {type(user)}")

    # 处理 base
    base = user.get('base') or user.get('_base')
    if base:
        if isinstance(base, str):
            base_obj = _load_json_from_path(base)
            if not isinstance(base_obj, dict):
                raise TypeError(f"Base file {base!r} must contain a JSON object, got {type(base_obj)}")
        elif isinstance(base, dict):
            base_obj = base
        else:
            raise TypeError(f"'base' must be path or dict, got {type(base)}")

        merged = json.loads(json.dumps(base_obj))
        user_wo_base = dict(user)
        user_wo_base.pop('base', None)
        user_wo_base.pop('_base', None)
        return _deep_merge(merged, user_wo_base)

    return user
