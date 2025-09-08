# -*- coding: utf-8 -*-
import json, os, subprocess, time, itertools, random
from copy import deepcopy

BASE_DSL = {
    "insert_midpoint": False,
    "no_backtrack": { "enabled": True, "epsilon": 0.002 },
    "speed_cap":    { "enabled": True, "d0": 0.9, "v_far": 1.20, "v_near": 0.28 },

    "horizon": 180,
    "dt": 0.08,

    "weights": {
        "state": [18, 18, 2.5, 0.1, 0.1],
        "control": [0.08, 0.08],
        "terminal_velocity": 30.0,
        "terminal_steer": 3.5
    },

    "u_rate_weight": 0.45,
    "terminal_scale": 2.2,
    "terminal_box": { "enabled": True, "half_sizes": [0.05, 0.05] },

    "constraints": {
        "a_min": -1.2, "a_max": 1.8,
        "delta_min": -0.50, "delta_max": 0.50,
        "v_min": 0.0, "v_max": 0.90
    },

    "risk": "med",
    "shield": { "mode": "hybrid", "weight": 8.0 }
}

# 生成一个任务 DSL
def make_task(start_xy=(0.0, 0.0), goal_xy=(2.0, 1.0),
              obstacle=((1.2, 0.5), 0.45),
              seed=None):
    random.seed(seed)
    dsl = deepcopy(BASE_DSL)
    sx, sy = start_xy
    gx, gy = goal_xy
    (cx, cy), r = obstacle

    dsl["start"] = [float(sx), float(sy), 0.0, 0.0, 0.0]
    dsl["goal"]  = [float(gx), float(gy), 0.0, 0.0, 0.0]
    dsl["obstacle"] = {"center": [float(cx), float(cy)], "radius": float(r)}

    # 安全性小检查：起点/终点与障碍的最小间距（> r+0.05）
    def ok(p):
        dx, dy = p[0]-cx, p[1]-cy
        return (dx*dx + dy*dy)**0.5 >= (r + 0.05)
    if not (ok(dsl["start"]) and ok(dsl["goal"])):
        # 不安全就把障碍向中心线偏移一点
        cx2 = cx + (random.random()*0.2 - 0.1)
        cy2 = cy + (random.random()*0.2 - 0.1)
        dsl["obstacle"]["center"] = [cx2, cy2]
    return dsl

def run_case(dsl_path, out_prefix):
    cmd = ["python", "-m", "sim.sim_runner", dsl_path, "--out", out_prefix, "--llm", "none"]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=False)

def main():
    os.makedirs("batch_out", exist_ok=True)
    os.makedirs("dsl_gen", exist_ok=True)

    # ① 网格组合（可自行增删）
    starts = [(0.0, 0.0), (0.0, 0.2), (-0.2, 0.0)]
    goals  = [(2.0, 1.0), (2.2, 0.9), (1.8, 1.1)]
    obstacles = [((1.2, 0.5), 0.45), ((1.1, 0.55), 0.40), ((1.3, 0.45), 0.50)]

    # ② 也可以再加几组随机（注释掉就不随机）
    rand_cases = []
    for k in range(4):
        sx = random.uniform(-0.2, 0.2)
        sy = random.uniform(-0.1, 0.1)
        gx = random.uniform(1.8, 2.4)
        gy = random.uniform(0.8, 1.2)
        cx = random.uniform(1.0, 1.4)
        cy = random.uniform(0.4, 0.6)
        r  = random.uniform(0.40, 0.55)
        rand_cases.append(((sx, sy), (gx, gy), ((cx, cy), r)))

    # 合并：网格 + 随机
    cases = list(itertools.product(starts, goals, obstacles)) + rand_cases

    for idx, case in enumerate(cases, 1):
        if isinstance(case[2], tuple) and isinstance(case[2][0], tuple):
            start_xy, goal_xy, obstacle = case
        else:
            # rand_cases 的结构
            start_xy, goal_xy, obstacle = case

        dsl = make_task(start_xy, goal_xy, obstacle, seed=idx)
        dsl_name = f"dsl_gen/task_{idx:02d}.json"
        with open(dsl_name, "w", encoding="utf-8") as f:
            json.dump(dsl, f, ensure_ascii=False, indent=2)

        out_prefix = f"batch_out/run_{idx:02d}"
        run_case(dsl_name, out_prefix)

        # 小憩，避免过快刷屏
        time.sleep(0.5)

if __name__ == "__main__":
    main()
