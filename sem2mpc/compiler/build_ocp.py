# -*- coding: utf-8 -*-
import casadi as ca
from typing import Dict, Any, Tuple, List

from compiler.ackermann_model import AckermannModel
from compiler.load_task import load_task
from compiler.shield import soft_barrier  # 已包含在 shield.py 中


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pad5(v: List[float]) -> List[float]:
    """补到 5 维 [x,y,theta,v,delta]"""
    v = list(v)
    if len(v) < 5:
        v = v + [0.0] * (5 - len(v))
    return v[:5]


def _expand_state_weights(w) -> List[float]:
    """state 权重允许 3/5 维，统一为 5 维"""
    if w is None:
        return [10.0, 10.0, 1.0, 0.1, 0.1]
    w = list(w)
    if len(w) == 3:
        w = w + [0.1, 0.1]
    elif len(w) < 5:
        w = w + [0.1] * (5 - len(w))
    return w[:5]


def _expand_control_weights(r) -> List[float]:
    """control 权重允许标量/向量，统一为 2 维 [a, delta]"""
    if r is None:
        return [0.05, 0.05]
    if isinstance(r, (int, float)):
        return [float(r), float(r)]
    r = list(r)
    if len(r) == 1:
        return [float(r[0]), float(r[0])]
    return [float(r[0]), float(r[1])]


def apply_risk(task: Dict[str, Any]) -> Dict[str, Any]:
    """风险自适应：半径/步长/终端权重/转角界"""
    risk = task.get('risk', 'med')
    obs = task.get('obstacle') or {}
    constraints = task.setdefault('constraints', {})

    if 'radius' not in obs:
        obs['radius'] = 0.35
    if 'center' not in obs:
        obs['center'] = [1.2, 0.5]
    task['obstacle'] = obs

    if 'horizon' not in task:
        task['horizon'] = 50
    if 'terminal_scale' not in task:
        task['terminal_scale'] = 3.0

    dmax = float(constraints.get('delta_max', 0.5))
    dmin = float(constraints.get('delta_min', -dmax))

    if risk == 'low':
        obs['radius'] = float(obs.get('radius', 0.35)) - 0.05
        task['horizon'] = int(_clamp(task.get('horizon', 50) - 10, 20, 120))
    elif risk == 'high':
        obs['radius'] = float(obs.get('radius', 0.35)) + 0.15
        task['horizon'] = int(_clamp(task.get('horizon', 50) + 20, 20, 120))
        task['terminal_scale'] = float(task.get('terminal_scale', 3.0)) * 1.2
        dmax = _clamp(dmax * 0.8, 0.15, 0.6)
        dmin = -dmax

    constraints['delta_max'] = dmax
    constraints['delta_min'] = dmin
    task['constraints'] = constraints
    return task


def _apply_goal_bias_and_side(task: Dict[str, Any]) -> Tuple[List[float], List[float], str]:
    """返回 (x0, xf, side)；支持 goal_bias 与 bias.side"""
    x0 = _pad5(task['start'])
    xf = _pad5(task['goal'])

    bias = task.get('goal_bias', None)
    if bias and len(bias) >= 2:
        xf[0] += float(bias[0]); xf[1] += float(bias[1])

    side = None
    if isinstance(task.get('bias'), dict):
        side = str(task['bias'].get('side')).lower() if task['bias'].get('side') else None
    return x0, xf, side


def build_ocp(task_or_path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    构建 OCP：
      等式：初值/动力学
      不等式：避障、控制/速度界、距离限速、不回退、终端硬盒
      代价：状态（含终端）、控制、控制变化率、终端速度/转角
    """
    task = apply_risk(load_task(task_or_path))

    # 权重
    w_state = _expand_state_weights(task.get('weights', {}).get('state'))
    w_ctrl  = _expand_control_weights(task.get('weights', {}).get('control'))
    terminal_scale = float(task.get('terminal_scale', 3.0))
    wdict = task.get('weights', {}) or {}
    q_vf = float(wdict.get('terminal_velocity', wdict.get('terminal_vel', 2.0)))
    q_df = float(wdict.get('terminal_steer', 1.0))
    u_rate_w = float(task.get('u_rate_weight', 0.0))

    # 模型/时间栅格
    model = AckermannModel()     # X=[x,y,theta,v,delta], U=[a, delta_cmd]
    nx, nu = model.nx, model.nu
    N  = int(task.get('horizon', 50))
    dt = float(task.get('dt', 0.1)); dt = _clamp(dt, 0.02, 0.3)

    # 初末状态与中点引导
    x0, xf, side = _apply_goal_bias_and_side(task)
    Q  = ca.diag(ca.DM(w_state))
    R  = ca.diag(ca.DM(w_ctrl))
    Qf = Q * terminal_scale

    X = ca.MX.sym('X', nx, N + 1)
    U = ca.MX.sym('U', nu, N)

    g_list: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []
    obj = 0

    # 初值
    g_list.append(X[:, 0] - ca.DM(x0)); lbg += [0.0]*nx; ubg += [0.0]*nx

    # via-points（Method C）
    if task.get('insert_midpoint', True):
        mid = [(x0[0]+xf[0])/2.0, (x0[1]+xf[1])/2.0, 0.0, 0.0, 0.0]
        if side == 'left':
            mid[0] -= 0.30; mid[1] += 0.20
        elif side == 'right':
            mid[0] += 0.30; mid[1] -= 0.20
        via_points = [mid, xf]
    else:
        via_points = [xf]

    # 机制开关：距离限速 + 不回退
    scfg = task.get("speed_cap", {}) or {}
    cap_enabled = bool(scfg.get("enabled", False))
    d0    = float(scfg.get("d0", 1.5))
    v_far = float(scfg.get("v_far", 0.8))
    v_near= float(scfg.get("v_near", 0.12))

    nbcfg = task.get("no_backtrack", {}) or {}
    nb_enabled = bool(nbcfg.get("enabled", False))
    eps_back   = float(nbcfg.get("epsilon", 0.002))

    goal_xy = ca.DM(xf[:2])

    # 动力学 & 阶段代价
    for k in range(N):
        xk = X[:, k]; uk = U[:, k]
        x_next = model.forward(xk, uk, dt)

        g_list.append(X[:, k+1] - x_next); lbg += [0.0]*nx; ubg += [0.0]*nx

        ref = via_points[0] if (len(via_points) > 1 and k < N//2) else via_points[-1]
        ref = ca.DM(ref)

        obj += ca.mtimes([(xk - ref).T, Q, (xk - ref)]) + ca.mtimes([uk.T, R, uk])

        if u_rate_w > 0 and k >= 1:
            du = U[:, k] - U[:, k-1]
            obj += u_rate_w * ca.mtimes(du.T, du)

        # 距离相关限速（对 v_k）
        if cap_enabled:
            pos_k = xk[0:2]
            d = ca.norm_2(goal_xy - pos_k)
            alpha = ca.fmin(1.0, d / d0)
            v_cap_k = v_near + (v_far - v_near) * alpha
            g_list.append(xk[3] - v_cap_k)   # xk[3] <= v_cap_k
            lbg.append(-ca.inf); ubg.append(0.0)

        # 不回退：沿目标方向 (p_{k+1}-p_k)·dir_to_goal >= -eps
        if nb_enabled:
            pos_k  = xk[0:2]
            pos_k1 = X[0:2, k+1]
            dir_goal = (goal_xy - pos_k) / (1e-6 + ca.norm_2(goal_xy - pos_k))
            forward_step = ca.dot(pos_k1 - pos_k, dir_goal)
            g_list.append(forward_step + eps_back)  # >= 0
            lbg.append(0.0); ubg.append(ca.inf)

    # 终端代价（位置 + 终端速度/转角）
    xN = X[:, -1]
    obj += ca.mtimes([(xN - ca.DM(xf)).T, Qf, (xN - ca.DM(xf))])
    obj += q_vf * (xN[3]**2) + q_df * (xN[4]**2)

    # 终端盒：硬约束（必须进入 ±eps_pos）
    eps_pos = float(task.get('terminal_box', {}).get('half_sizes', [0.05, 0.05])[0])
    eps_pos = _clamp(eps_pos, 0.03, 0.20)
    g_list += [
        (X[0, -1] - xf[0]) - eps_pos,
        -(X[0, -1] - xf[0]) - eps_pos,
        (X[1, -1] - xf[1]) - eps_pos,
        -(X[1, -1] - xf[1]) - eps_pos,
    ]
    lbg += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
    ubg += [0.0, 0.0, 0.0, 0.0]

    # 避障：硬/软/混合
    obs = task.get('obstacle', None)
    if obs:
        cx = float(obs['center'][0]); cy = float(obs['center'][1])
        r = float(obs['radius']); r2 = r * r
        shield_cfg = task.get('shield', {}) or {}
        mode = str(shield_cfg.get('mode', 'hard')).lower()
        soft_w = float(shield_cfg.get('weight', 8.0))
        use_hard = mode in ['hard', 'hybrid']
        use_soft = mode in ['soft', 'hybrid']
        for k in range(N + 1):
            dx = X[0, k] - cx; dy = X[1, k] - cy; dist2 = dx*dx + dy*dy
            if use_hard:
                g_list.append(dist2 - r2); lbg.append(0.0); ubg.append(float('inf'))
            if use_soft:
                obj += soft_barrier(dist2, r2, w=soft_w)

    # 控制上下界
    cons = task.get('constraints', {}) or {}
    a_min = float(cons.get('a_min', -1.0)); a_max = float(cons.get('a_max', +1.0))
    d_min = float(cons.get('delta_min', -0.5)); d_max = float(cons.get('delta_max', +0.5))

    umin = ca.DM([a_min, d_min]); umax = ca.DM([a_max, d_max])
    for k in range(N):
        g_list.append(U[:, k] - umin); lbg += [0.0]*nu; ubg += [float('inf')]*nu
        g_list.append(umax - U[:, k]); lbg += [0.0]*nu; ubg += [float('inf')]*nu

    # 速度上下界（若提供）
    v_min = cons.get('v_min', None); v_max = cons.get('v_max', None)
    if v_min is not None:
        vmin = float(v_min)
        for k in range(N + 1):
            g_list.append(X[3, k] - vmin); lbg.append(0.0); ubg.append(float('inf'))
    if v_max is not None:
        vmax = float(v_max)
        for k in range(N + 1):
            g_list.append(vmax - X[3, k]); lbg.append(0.0); ubg.append(float('inf'))

    # 打包 NLP
    vars_ = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g = ca.vertcat(*g_list)
    nlp = {'x': vars_, 'f': obj, 'g': g}

    meta = {
        'N': N, 'nx': nx, 'nu': nu,
        'obstacle': obs,
        'bounds': {'lbg': lbg, 'ubg': ubg},
    }
    return nlp, meta
