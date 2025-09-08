# sem2mpc/semantics/providers/multi_model_provider.py
import requests, json, time, os, random
from typing import List, Optional, Tuple, Callable

def _one_call_ollama(prompt:str, model:str, base_url:str, temperature:float,
                     num_predict:int, format_json:bool, seed:Optional[int], timeout:int, debug_dir:Optional[str]) -> Tuple[bool, str]:
    endpoint = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json" if format_json else None,
        "options": {
            "temperature": float(temperature),
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "num_predict": int(num_predict)
        }
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)

    r = requests.post(endpoint, json=payload, timeout=timeout)
    ok = r.status_code == 200
    text = r.text
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        ts = int(time.time()*1000)
        with open(os.path.join(debug_dir, f"ollama_{model}_{ts}.json"), "w", encoding="utf-8") as f:
            f.write(text)
    if not ok:
        return False, f"HTTP {r.status_code}: {text}"
    try:
        data = r.json()
        resp = (data.get("response") or "").strip()
        return True, resp
    except Exception as e:
        return False, f"JSON parse error: {e}, raw={text[:2000]}"

def _try_parse_json(s: str):
    try:
        return True, json.loads(s)
    except Exception:
        return False, None

def make_multi_model_provider(
    models: List[str],
    base_url: str = "http://127.0.0.1:11434",
    k_samples: int = 1,
    temperature: float = 0.0,
    num_predict: int = 256,
    timeout: int = 120,
    format_json: bool = True,
    seed: Optional[int] = 42,
    debug_dir: Optional[str] = "llm_logs"
) -> Callable[[str], str]:
    """
    返回 provider(prompt)->JSON字符串
    - 支持多模型按顺序尝试
    - 支持每个模型做 k 次采样（自一致性 majority vote）
    - 强制 JSON（format=json）+ 0温度默认为确定性
    """
    models = list(models)

    def provider(prompt: str) -> str:
        best_json = None
        # 收集可解析 JSON 的候选（做简单多数投票）
        votes = {}

        for m in models:
            for i in range(max(1, k_samples)):
                sd = None if seed is None else seed + i
                ok, resp = _one_call_ollama(
                    prompt=prompt, model=m, base_url=base_url,
                    temperature=temperature, num_predict=num_predict,
                    format_json=format_json, seed=sd, timeout=timeout,
                    debug_dir=debug_dir
                )
                if not ok:
                    continue
                okj, obj = _try_parse_json(resp)
                if not okj or not isinstance(obj, dict):
                    continue
                # 归一化：转成紧凑字符串帮忙vote
                key = json.dumps(obj, sort_keys=True)
                votes[key] = votes.get(key, 0) + 1
                if best_json is None:
                    best_json = obj

        if not votes:
            # 全部失败：返回一个空 patch 让上层 fallback
            return "{}"

        # 取票数最高的 JSON
        winner = max(votes.items(), key=lambda kv: kv[1])[0]
        return winner  # 已是 JSON 字符串

    return provider
