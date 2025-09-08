# run.py  (放在仓库根目录)
# -*- coding: utf-8 -*-
import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

BANNER = """
==============================
  SafeTalk-MPC Interactive
==============================
说明:
- 直接输入自然语言(如: 快一些, 他要赶飞机) 或 粘贴 JSON 补丁
- 输入 'exit' / 'quit' 退出
- 输入 'help' 查看帮助, 输入 'env' 查看关键环境变量状态
"""

HELP_TEXT = """
(帮助)
- 直接输入中文口令: 例 "要稳一点, 车上有小孩老人"
- 或者输入 JSON 补丁: 例 {"constraints":{"v_max":1.10,"a_max":1.60},"speed_cap":{"v_far":1.30,"v_near":0.35}}
- 特殊命令:
    exit / quit : 退出
    help        : 显示帮助
    env         : 显示 OPENAI_API_KEY / DASHSCOPE_API_KEY 是否可见
- 结果会保存到 exp/RUN_<timestamp>/ 下, 包括 *_result.png、*_metrics.json、last_patch.json 等
"""

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeTalk-MPC Interactive Runner")
    p.add_argument("--task", default="", help="DSL JSON path; 留空则自动探测")
    p.add_argument("--llm", default="ollama", choices=["none", "ollama", "openai", "dashscope"],
                   help="LLM backend (default: ollama)")
    p.add_argument("--model", default="qwen2.5:7b",
                   help="Model name. ollama: qwen*/llama*; openai: gpt-4o(-mini)/gpt-4.1; dashscope: qwen-plus 等")
    p.add_argument("--base-url", default="http://127.0.0.1:11434",
                   help="Base URL. ollama: http://127.0.0.1:11434; openai: https://api.openai.com; dashscope: 忽略")
    p.add_argument("--save-llm", action="store_true", help="Save raw LLM logs to llm_logs/")
    return p.parse_args()

def autodetect_task(user_task: str) -> str:
    """优先 sem2mpc/dsl/base.json → dsl/base.json；若用户给了就用用户的。"""
    if user_task:
        return user_task
    root = Path(__file__).parent
    cand = [
        root / "sem2mpc" / "dsl" / "base.json",
        root / "dsl" / "base.json",
    ]
    for p in cand:
        if p.exists():
            return str(p.as_posix())
    # 允许传内联 JSON（以 { 开头），否则提示一下
    print("⚠️ 警告: 未找到默认 DSL 文件。你可以用 --task 指定路径，或在提示中直接粘贴内联 JSON 作为 DSL（需要以 '{' 开头）。")
    return "dsl/base.json"

def make_outdir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path("exp") / f"RUN_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def visible_key(name: str) -> str:
    v = os.getenv(name)
    if not v:
        return f"{name}=<not set>"
    return f"{name}=<{'*'*4}{v[-4:]}>"

def print_env_hint(llm: str, base_url: str):
    print(f"[配置] LLM={llm}  base-url={base_url}")
    if llm == "openai":
        print("  注意: 使用 OpenAI 兼容接口需要设置环境变量 OPENAI_API_KEY")
    if llm == "dashscope":
        print("  注意: 使用 DashScope 需要设置环境变量 DASHSCOPE_API_KEY")

def run_once(task: str, instr: str, llm: str, model: str, base_url: str, save_llm: bool, outdir: Path) -> int:
    cmd = [
        sys.executable, "-m", "sem2mpc.sim.sim_runner",   # ← 更稳妥：显式带包名前缀
        task, instr,
        "--llm", llm,
        "--model", model,
        "--base-url", base_url,
        "--out", str(outdir),
    ]
    if save_llm:
        cmd.append("--save-llm")

    print("\n[命令] " + " ".join(shlex.quote(c) for c in cmd))
    print("------------------------------------------------------------")
    proc = subprocess.run(cmd)
    print("------------------------------------------------------------\n")
    return proc.returncode

def main():
    args = parse_args()
    print(BANNER)
    print_env_hint(args.llm, args.base_url)
    print(HELP_TEXT)

    task = autodetect_task(args.task)
    # 如果不是内联 JSON（以 { 开头），且文件不存在，就提示
    if not task.lstrip().startswith("{") and not Path(task).exists():
        print(f"⚠️ 警告: 未找到任务文件 '{task}'。你可以用 --task 指定完整路径，或在输入中粘贴内联 JSON 作为 DSL。")

    while True:
        try:
            instr = input("📝 请输入自然语言 / JSON 补丁 (exit 退出)> ").strip()
        except KeyboardInterrupt:
            print("\n(按 Ctrl+C 退出)")
            break

        if not instr:
            continue
        lower = instr.lower()
        if lower in ("exit", "quit"):
            print("再见 👋")
            break
        if lower == "help":
            print(HELP_TEXT)
            continue
        if lower == "env":
            print(visible_key("OPENAI_API_KEY"))
            print(visible_key("DASHSCOPE_API_KEY"))
            continue

        outdir = make_outdir()
        code = run_once(task, instr, args.llm, args.model, args.base_url, args.save_llm, outdir)
        if code == 0:
            print(f"✅ 完成。结果保存在: {outdir}")
        else:
            print(f"❌ 运行失败，返回码: {code}")

if __name__ == "__main__":
    main()
