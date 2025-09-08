# sem2mpc/semantics/providers/ollama_provider.py
# -*- coding: utf-8 -*-
"""
Ollama provider: 强制 JSON 输出 + 兼容 /api/generate 和 /api/chat
"""

from __future__ import annotations
import json, os, time, uuid, re
from typing import Callable, Optional, Dict, Any
import requests

JSON_EXTRACT = re.compile(r"\{.*\}", re.S)

SYSTEM = (
    "你是一个“语义到MPC补丁”的编译器。只输出一个 JSON 对象，不要解释、不要前后缀、不要 Markdown 代码块。"
    "把中文口语映射到 DSL 可调键："
    "constraints.v_max, constraints.a_max, speed_cap.v_far, speed_cap.v_near, "
    "weights.tracking, weights.control, weights.smooth, u_rate_weight, obstacle.radius。"
    "如果用户给了具体数字（如 v_max=1.10 或 安全半径0.6），优先采用该数字；否则根据语义做小幅调整，数值保持合理范围。"
    "示例输出："
    "{\"constraints\":{\"v_max\":1.10,\"a_max\":1.60},\"speed_cap\":{\"v_far\":1.30,\"v_near\":0.35}}"
)

USER_TMPL = (
    "用户指令：{instr}\n"
    "当前可调DSL摘要：{dsl_hint}\n"
    "只输出JSON对象。"
)

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _save_debug(save_dir: Optional[str], name: str, payload: Any):
    if not save_dir:
        return
    _ensure_dir(save_dir)
    fp = os.path.join(save_dir, f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{name}.txt")
    try:
        with open(fp, "w", encoding="utf-8") as f:
            if isinstance(payload, (dict, list)):
                f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                f.write(str(payload))
    except Exception:
        pass

def _extract_json(text: str) -> str:
    m = JSON_EXTRACT.search(text or "")
    if m:
        return m.group(0)
    # 没匹配到，直接返回原文（上层会再尝试）
    return text or ""

def make_ollama_provider(
    model: str,
    base_url: str,
    temperature: float = 0.0,
    num_predict: int = 256,
    seed: int = 42,
    save_dir: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 2,
) -> Callable[[str, Optional[str], Optional[Dict[str, Any]]], str]:
    """
    返回一个 provider(instruction, dsl_hint, context)->str
    - 优先调用 /api/generate，并设置 format=json 强制 JSON
    - 如果 generate 不可用，退回 /api/chat
    - 始终把原始返回保存到 save_dir 便于排查
    """
    base_url = base_url.rstrip("/")

    def _post_json(url: str, payload: dict) -> dict:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _call_generate(prompt: str) -> str:
        url = f"{base_url}/api/generate"
        # Ollama generate 支持 format='json'（新版本），老版本忽略该字段不报错
        payload = {
            "model": model,
            "prompt": prompt,
            "system": SYSTEM,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(num_predict),
                "seed": int(seed),
            },
            "format": "json",
            "stream": False,
        }
        _save_debug(save_dir, "ollama_req_generate", payload)
        resp = _post_json(url, payload)
        _save_debug(save_dir, "ollama_resp_generate", resp)
        text = resp.get("response") or resp.get("text") or ""
        return text

    def _call_chat(prompt: str) -> str:
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": float(temperature),
                "num_predict": int(num_predict),
                "seed": int(seed),
            },
            "stream": False,
            # 注意：部分 Ollama 版本也支持 format='json'，这里也试一下；不支持会被忽略
            "format": "json",
        }
        _save_debug(save_dir, "ollama_req_chat", payload)
        resp = _post_json(url, payload)
        _save_debug(save_dir, "ollama_resp_chat", resp)
        # chat API 返回结构可能有 message / content
        msg = (resp.get("message") or {}).get("content") or resp.get("content") or ""
        return msg

    def _provider(instruction: str, dsl_hint: Optional[str], ctx: Optional[Dict[str, Any]]) -> str:
        prompt = USER_TMPL.format(instr=instruction, dsl_hint=dsl_hint or "{}")
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                try:
                    text = _call_generate(prompt)
                except Exception as e1:
                    last_err = e1
                    text = _call_chat(prompt)
                if save_dir:
                    _save_debug(save_dir, "ollama_raw_text", text)
                # 尽力抽 JSON
                j = _extract_json(text)
                # 提前验证是否真的是 JSON
                try:
                    json.loads(j)
                    return j
                except Exception:
                    # 再退一步：如果模型输出已经是一个 dict 风格字符串，但包裹了注释或额外文本，抽取正则可能没吃全
                    # 直接返回原文，上层 semantic_compiler 还会再尝试一次 _extract_jsonish_block
                    return text
            except Exception as e:
                last_err = e
        # 连续失败，抛最后一次错误
        raise RuntimeError(f"Ollama provider failed: {last_err}")

    return _provider
