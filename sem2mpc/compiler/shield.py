# -*- coding: utf-8 -*-
from typing import Dict, Any
import casadi as ca

# --- 既有：软障碍代价（与 build_ocp 兼容） ---
def soft_barrier(dist2, r2, w=8.0, eps=1e-6):
    """
    对 dist^2 < r^2 的违规区域施加平滑惩罚：w * (max(0, r^2 - dist^2))^2
    """
    return w * ca.fmax(0, (r2 - dist2) + eps)**2

# --- 新增：Safety Shield 的补丁裁剪 ---
def _clip(x, lo, hi): 
    return max(lo, min(hi, x))

def sanitize_patch(p: Dict[str, Any]) -> Dict[str, Any]:
    """对 LLM 生成的补丁做安全裁剪/保底，避免不可行或危险设置"""
    p = dict(p)  # 浅拷贝

    # 1) 安全半径下限
    obs = p.get("obstacle")
    if isinstance(obs, dict):
        r = obs.get("radius")
        if isinstance(r, (int, float)):
            obs["radius"] = _clip(float(r), 0.35, 1.50)  # 可按需要调整上下限
        p["obstacle"] = obs

    # 2) 控制/速度边界
    cons = p.get("constraints", {})
    if isinstance(cons, dict):
        if "v_max" in cons and isinstance(cons["v_max"], (int, float)):
            cons["v_max"] = _clip(float(cons["v_max"]), 0.3, 1.5)
        if "a_max" in cons and isinstance(cons["a_max"], (int, float)):
            cons["a_max"] = _clip(float(cons["a_max"]), 0.5, 2.5)
        if "delta_max" in cons and isinstance(cons["delta_max"], (int, float)):
            dmax = _clip(float(cons["delta_max"]), 0.2, 0.7)
            cons["delta_max"] = dmax
            cons["delta_min"] = -dmax
        p["constraints"] = cons

    # 3) 终端盒范围（太紧会 infeasible）
    tbox = p.get("terminal_box")
    if isinstance(tbox, dict):
        hs = tbox.get("half_sizes")
        if isinstance(hs, list) and len(hs) >= 2:
            tbox["half_sizes"][0] = _clip(float(hs[0]), 0.04, 0.12)
            tbox["half_sizes"][1] = _clip(float(hs[1]), 0.04, 0.12)
        p["terminal_box"] = tbox

    # 4) 预测时域（防止 LLM 乱设）
    if "horizon" in p and isinstance(p["horizon"], (int, float)):
        p["horizon"] = int(_clip(int(p["horizon"]), 60, 220))

    return p
