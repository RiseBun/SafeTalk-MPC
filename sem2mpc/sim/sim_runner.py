# sim/sim_runner.py
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
SafeTalk-MPC - simulation runner

功能：
- 读取 DSL(JSON) -> 调用 build_ocp 构建 NLP
- 自动根据 nlp['x'] 的真实长度构造初值
- 求解并绘图/动画，保存指标
- 支持两种补丁方式：本地 JSON / LLM 编译器
- 包方式（python -m sem2mpc.sim.sim_runner）与脚本方式（cd sem2mpc; python -m sim.sim_runner）均可运行
"""

import os
import sys
import json
import time
import csv
import argparse
import platform
import inspect
import numpy as np
import casadi as ca

# ✅ 无界面环境也能出图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 统一导入 shim：包方式优先，脚本方式回退 ---
try:
    # 包方式
    from sem2mpc.compiler.build_ocp import build_ocp
    from sem2mpc.sim.plot_animation import plot_trajectory_animation
    from sem2mpc.semantics.semantic_compiler import compile_from_text as _compile_from_text
except Exception:
    # 脚本方式
    from compiler.build_ocp import build_ocp
    from sim.plot_animation import plot_trajectory_animation
    from semantics.semantic_compiler import compile_from_text as _compile_from_text

# 绑定名字，避免 NameError
compile_from_text = _compile_from_text

# --- Provider shim ---
_mk_ollama = None
_mk_openai = None
_mk_dashscope = None

# Ollama
try:
    from sem2mpc.semantics.providers.ollama_provider import make_ollama_provider as _mk_ollama
except Exception:
    try:
        from semantics.providers.ollama_provider import make_ollama_provider as _mk_ollama
    except Exception:
        _mk_ollama = None

# OpenAI
try:
    from sem2mpc.semantics.providers.openai_provider import make_openai_provider as _mk_openai
except Exception:
    try:
        from semantics.providers.openai_provider import make_openai_provider as _mk_openai
    except Exception:
        _mk_openai = None

# DashScope
try:
    from sem2mpc.semantics.providers.dashscope_provider import make_dashscope_provider as _mk_dashscope
except Exception:
    try:
        from semantics.providers.dashscope_provider import make_dashscope_provider as _mk_dashscope
    except Exception:
        _mk_dashscope = None



# -------------------------
# 工具函数
# -------------------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _save_json(obj, path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _ensure_meta(nlp, meta):
    if isinstance(meta, dict) and 'N' in meta:
        if 'bounds' not in meta or 'lbg' not in meta['bounds'] or 'ubg' not in meta['bounds']:
            ng = int(nlp['g'].size1())
            meta.setdefault('bounds', {})
            meta['bounds']['lbg'] = [0.0] * ng
            meta['bounds']['ubg'] = [1e9] * ng
        return meta
    try:
        N, nx, nu = meta
    except Exception:
        raise RuntimeError("build_ocp 必须返回 (nlp, meta)，其中 meta 至少包含 {N, nx, nu} 或等价元组。")
    ng = int(nlp['g'].size1())
    return {'N': int(N), 'nx': int(nx), 'nu': int(nu),
            'bounds': {'lbg': [0.0]*ng, 'ubg': [1e9]*ng}, 'obstacle': None}

def _jsonable_patch(patch: dict) -> dict:
    out = {}
    for k, v in (patch or {}).items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out

def _append_csv_log(csv_path: str, row: dict):
    header = list(row.keys())
    exists = os.path.isfile(csv_path)
    _ensure_dir(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

def _looks_like_json_text(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.lstrip()
    return t.startswith("{") or t.startswith("[")

def _is_json_path(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip().strip('"').strip("'")
    return t.lower().endswith(".json") or ("/" in t) or ("\\" in t)

def _load_patch_from_arg(arg: str):
    if _looks_like_json_text(arg):
        try:
            return json.loads(arg)
        except Exception as e:
            raise ValueError(f"Inline JSON patch is invalid: {e}")
    if _is_json_path(arg):
        p = arg.strip().strip('"').strip("'")
        if os.path.isfile(p):
            return _load_json(p)
        abs_p = os.path.abspath(p)
        if os.path.isfile(abs_p):
            return _load_json(abs_p)
        raise FileNotFoundError(f"Patch file not found: {arg} (abs: {abs_p})")
    raise ValueError("When --llm none, the second argument must be a JSON file path or an inline JSON object.")

def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


# -------------------------
# 🔎 差异报告（更稳）：直接比较 base vs patched
# -------------------------
_NUM_KEYS_HINT = {"constraints", "speed_cap", "weights", "obstacle", "u_rate_weight"}

def _dict_diff(a, b, prefix=""):
    diffs = []
    a = a if isinstance(a, dict) else {}
    b = b if isinstance(b, dict) else {}
    keys = set(a.keys()) | set(b.keys())
    for k in sorted(keys):
        p = f"{prefix}.{k}" if prefix else k
        av, bv = a.get(k, None), b.get(k, None)
        if isinstance(av, dict) or isinstance(bv, dict):
            diffs += _dict_diff(av if isinstance(av, dict) else {}, bv if isinstance(bv, dict) else {}, p)
        else:
            if av != bv:
                diffs.append((p, av, bv))
    return diffs

def _print_diff_explanation(base_task: dict, patched_task: dict):
    diffs = _dict_diff(base_task, patched_task)
    if not diffs:
        print("ℹ️ 本次未检测到 DSL 参数变化（可能 LLM/回退未生成补丁）。")
        return
    print("🔎 本次语义指令导致的 DSL 参数改动：")
    for p, old, newv in diffs:
        # 只打印和 MPC 相关的常见键（可按需放宽）
        if p.split(".")[0] in _NUM_KEYS_HINT:
            try:
                old_s = json.dumps(old, ensure_ascii=False)
                new_s = json.dumps(newv, ensure_ascii=False)
            except Exception:
                old_s, new_s = str(old), str(newv)
            print(f" - {p}: {old_s} → {new_s}")


# -------------------------
# 求解 + 作图
# -------------------------
def solve_and_plot(task_json_path, out_prefix='mpc'):
    print('🛠️ Building MPC problem from DSL...')
    build_ret = build_ocp(task_json_path)
    if isinstance(build_ret, (list, tuple)) and len(build_ret) == 2:
        nlp, meta = build_ret
    else:
        raise RuntimeError("build_ocp 必须返回 (nlp, meta)")

    meta = _ensure_meta(nlp, meta)
    N, nx, nu = meta['N'], meta['nx'], meta['nu']
    lbg, ubg = meta['bounds']['lbg'], meta['bounds']['ubg']

    solver = ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level': 0, 'print_time': 0,
        'ipopt.max_iter': 400, 'ipopt.tol': 1e-6
    })

    n_dec = int(nlp['x'].size1())
    base_len = nx * (N + 1) + nu * N
    extra = n_dec - base_len
    if extra < 0:
        raise RuntimeError(f"Internal error: decision size smaller than X/U block. n_dec={n_dec}, base_len={base_len}")

    x_init = ca.DM.zeros((nx, N + 1))
    u_init = ca.DM.zeros((nu, N))
    init_guess = ca.vertcat(ca.reshape(x_init, -1, 1), ca.reshape(u_init, -1, 1))
    if extra > 0:
        init_guess = ca.vertcat(init_guess, ca.DM.zeros(extra, 1))

    print('🚀 Solving MPC...')
    t0 = time.time()
    sol = solver(x0=init_guess, lbg=lbg, ubg=ubg)
    t1 = time.time()

    if 'x' not in sol:
        raise RuntimeError("❌ IPOPT 未返回解向量 x")

    x_opt = ca.reshape(sol['x'][:nx * (N + 1)], nx, N + 1)
    xs = x_opt.T.full()
    xs_xy = xs[:, :2]

    task_cfg = _load_json(task_json_path)
    goal = np.array(task_cfg.get('goal', [2, 1, 0, 0, 0]))[:2]
    end_err = float(np.linalg.norm(xs_xy[-1] - goal))
    tot_time = t1 - t0

    obstacle = None
    min_dist = None
    if meta.get('obstacle'):
        cx = float(meta['obstacle']['center'][0])
        cy = float(meta['obstacle']['center'][1])
        r = float(meta['obstacle']['radius'])
        d = np.sqrt((xs_xy[:, 0] - cx) ** 2 + (xs_xy[:, 1] - cy) ** 2)
        min_dist = float(np.min(d))
        obstacle = (cx, cy, r)

    plt.figure(figsize=(6, 6))
    plt.plot(xs_xy[:, 0], xs_xy[:, 1], 'b-o', ms=3, label='trajectory')
    plt.scatter([xs_xy[0, 0]], [xs_xy[0, 1]], c='g', s=60, label='start')
    plt.scatter([goal[0]], [goal[1]], c='r', s=60, label='goal')
    if obstacle:
        circ = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.35, label='obstacle')
        plt.gca().add_patch(circ)
    plt.axis('equal'); plt.grid(True); plt.legend(); plt.title('SafeTalk-MPC Trajectory')
    fig_path = f"{out_prefix}_result.png"
    _ensure_dir(fig_path)
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"📷 saved {fig_path}")

    if xs.shape[0] >= 3:
        try:
            anim_path = f"{out_prefix}_anim.mp4"
            plot_trajectory_animation(xs[:, :3], anim_path, obstacle=obstacle or (1.0, 0.5, 0.3))
            print(f"🎥 saved {anim_path}")
        except Exception as e:
            print(f"⚠️ 动画导出失败（忽略）：{e}")

    metrics = {'N': int(N),
               'end_position_error': end_err,
               'min_obstacle_distance': min_dist,
               'solve_time_sec': tot_time}
    _save_json(metrics, f"{out_prefix}_metrics.json")
    print("📑 metrics:", metrics)
    return metrics


# -------------------------
# Provider 构造
# -------------------------
def _make_provider(args):
    if args.llm == 'ollama':
        if _mk_ollama is None:
            return None
        return _mk_ollama(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temp,
            num_predict=256,
            seed=args.seed,
            save_dir="llm_logs" if args.save_llm else None,
            timeout=120,
            max_retries=2,
        )
    if args.llm == 'openai':
        if _mk_openai is None:
            print("⚠️ openai provider 不可用（未找到模块 semantics.providers.openai_provider）")
            return None
        # API Key 从环境变量读取
        return _mk_openai(
            model=args.model,
            api_base=args.base_url,
            api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=args.temp,
            save_dir="llm_logs" if args.save_llm else None,
            timeout=120,
            max_retries=2,
        )
    if args.llm == 'dashscope':
        if _mk_dashscope is None:
            print("⚠️ dashscope provider 不可用（未找到模块 semantics.providers.dashscope_provider）")
            return None
        return _mk_dashscope(
            model=args.model,
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            temperature=args.temp,
            save_dir="llm_logs" if args.save_llm else None,
            timeout=120,
            max_retries=2,
        )
    return None


# -------------------------
# CLI
# -------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="SafeTalk-MPC simulation runner")
    p.add_argument('task', nargs='?', default='dsl/example_task_curve_01.json',
                   help='DSL JSON path (default: dsl/example_task_curve_01.json)')
    p.add_argument('instruction', nargs='?', default=None,
                   help='Natural language instruction / JSON patch / patch file path')
    p.add_argument('--out', default='mpc', help='output file prefix (default: mpc)')

    # LLM 相关
    p.add_argument('--llm', default='ollama', choices=['none', 'ollama', 'openai', 'dashscope'],
                   help='LLM backend (default: ollama)')
    p.add_argument('--model', default='qwen2.5:7b-instruct', help='model name (ollama: qwen/llama; openai: gpt-4o-mini 等)')
    p.add_argument('--k', type=int, default=1, help='(reserved) samples per model (default: 1)')
    p.add_argument('--temp', type=float, default=0.0, help='LLM temperature (default: 0.0)')
    p.add_argument('--seed', type=int, default=42, help='sampling seed (default: 42)')

    # URL：对 ollama 是 http://127.0.0.1:11434；对 openai 可填自定义反代；对 dashscope 忽略
    p.add_argument('--base-url', default='http://127.0.0.1:11434', help='Ollama/OpenAI base URL')
    p.add_argument('--save-llm', action='store_true', help='save raw LLM logs to llm_logs/')
    p.add_argument('--risk', default='high', choices=['low', 'med', 'high'],
                   help='risk hint for semantics (default: high)')
    return p


def main():
    args = build_arg_parser().parse_args()
    task_path = args.task
    out_prefix = args.out
    instruction = args.instruction

    # ===== 分支 1：--llm none，本地补丁 =====
    if instruction is not None and args.llm == 'none':
        print(f"🗂️ Local patch mode (--llm none). Patch arg: {instruction}")
        patch_obj = _load_patch_from_arg(instruction)
        base_task = _load_json(task_path) if os.path.isfile(task_path) else json.loads(task_path)
        patched_task = _deep_merge(json.loads(json.dumps(base_task)), patch_obj)
        _save_json(_jsonable_patch(patch_obj), "last_patch.json")
        _save_json(patched_task, "_tmp_task.json")
        print("🧩 Local patch 已保存：last_patch.json")
        print("🧾 Patched DSL 已保存：_tmp_task.json")

        # ✅ 差异报告（不依赖 patch_obj）
        _print_diff_explanation(base_task, patched_task)

        metrics = solve_and_plot("_tmp_task.json", out_prefix=out_prefix)

    # ===== 分支 2：LLM 语义编译（ollama/openai/dashscope） =====
    elif instruction is not None and args.llm != 'none':
        print(f"🗣️ Instruction: {instruction}")

        provider = _make_provider(args)
        original_task = _load_json(task_path) if os.path.isfile(task_path) else task_path

        try:
            patched_task, patch = compile_from_text(
                original_task, instruction,
                context={'risk_hint': args.risk},
                provider=provider
            )
        except Exception as e:
            print(f"⚠️ LLM 编译失败，改用本地规则回退：{e}")
            patched_task, patch = compile_from_text(
                original_task, instruction,
                context={'risk_hint': args.risk},
                provider=None
            )

        _save_json(_jsonable_patch(patch), "last_patch.json")
        _save_json(patched_task, "_tmp_task.json")
        print("🧩 LLM patch 已保存：last_patch.json")
        print("🧾 Patched DSL 已保存：_tmp_task.json")
                # 🔖 打印补丁来源/规则/强度
        origin = patch.get("_origin", "unknown")
        rules  = patch.get("_rules", [])
        inten  = patch.get("_intensity", "")
        print(f"📌 补丁来源: {origin}" + (f"  (规则: {', '.join(rules)}; 强度: {inten})" if rules or inten else ""))


        # ✅ 差异报告（最稳）：直接对比 base vs patched
        base_task_dict = original_task if isinstance(original_task, dict) else _load_json(original_task)
        _print_diff_explanation(base_task_dict, patched_task)

        metrics = solve_and_plot("_tmp_task.json", out_prefix=out_prefix)

    # ===== 分支 3：无补丁 =====
    else:
        print("🗣️ No instruction. Run base task.")
        metrics = solve_and_plot(task_path, out_prefix=out_prefix)
    
    # （可选）把来源写进 CSV
    try:
        origin = patch.get("_origin", "")
        rules = ",".join(patch.get("_rules", []))
        inten = patch.get("_intensity", "")
    except NameError:
        origin = rules = inten = ""
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm": args.llm,
        "model": args.model,
        "temp": args.temp,
        "seed": args.seed,
        "instruction": instruction or "",
        "task": task_path,
        "end_err": metrics.get("end_position_error"),
        "min_dist": metrics.get("min_obstacle_distance"),
        "solve_time": metrics.get("solve_time_sec"),
        "N": metrics.get("N"),
        "machine": platform.platform(),
        "origin": origin,
        "rules": rules,
        "intensity": inten,
    }
    try:
        _append_csv_log("llm_runs.csv", row)
        print("🧾 appended log to llm_runs.csv")
    except Exception as e:
        print(f"⚠️ CSV 记录失败（忽略）：{e}")

    print('✅ Done.')


if __name__ == '__main__':
    main()
