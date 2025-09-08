# sem2mpc/semantics/llm_agent.py
import re, math, json

def _to_rad(val, unit):
    unit = (unit or "").lower()
    if unit in ["deg", "degree", "degrees", "度"]:
        return float(val) * math.pi / 180.0
    return float(val)

class LLMInterpreter:
    """
    把 (自然语言, 场景上下文) -> 对 DSL 的“增量修改 patch”
    结构：{ "key.path": (op, value, ...), ... }
    op 支持：
      - ("set", v)         : 直接赋值
      - ("add", x)         : 加法（标量）
      - ("mul", r)         : 乘法（标量/向量均可；向量逐元素 * r）
      - ("add_int", k)     : 加整数
      - ("scale", r)       : 专门用于 terminal_scale *= r
      - ("mul_clip", r, lo, hi) : 乘后夹到[lo,hi]
      - ("sym_from", key)  : 取对称值：delta_min = -delta_max
      - ("vec2", a, b)     : 直接把 2 维向量写入（control 权重用）
    """
    def __init__(self, provider=None):
        self.provider = provider

    # —— 强化版本地规则（LLM 不可用时也能看到显著差异）——
    def _fallback_rules(self, text, context):
        patch = {"_hits": []}
        t = text.lower()

        # A.“左右绕行”意图 —— 明显位移参考点（需要编译器端支持对 mid/goal 偏置）
        if re.search(r"(靠左|left)", t):
            patch["bias.side"] = ("set", "left")      # 语义标记，可被 build_ocp 使用
            patch["goal_bias"] = ("set", [-0.5, 0.2]) # 明显左偏
            patch["_hits"].append("side_left")

        if re.search(r"(靠右|right)", t):
            patch["bias.side"] = ("set", "right")
            patch["goal_bias"] = ("set", [0.5, -0.2]) # 明显右偏
            patch["_hits"].append("side_right")

        # B.“尽快/更快”——减小 R、缩短预测、增大步频；更容易“拼速度”
        if re.search(r"(尽快|更快|asap|quick|faster)", t):
            patch["weights.control"] = ("mul", 0.2)         # R * 0.2（大幅放松控制代价）
            patch["horizon"] = ("add_int", -30)             # N -= 30（最少仍由下游限幅）
            patch["dt"] = ("mul_clip", 0.8, 0.05, 0.2)      # dt *= 0.8
            patch["terminal_scale"] = ("scale", 0.7)        # 终端软化，促进行进
            patch["_hits"].append("faster")

        # C.“更平稳/舒适”——增大 R、加长预测、控制变化率更重
        if re.search(r"(更平稳|舒适|smooth|comfort|稳)", t):
            patch["weights.control"] = ("mul", 3.0)         # R * 3（强平滑）
            patch["horizon"] = ("add_int", +30)             # N += 30
            patch["u_rate_weight"] = ("set", 0.5)           # 新增：控制变化率项
            patch["terminal_scale"] = ("scale", 1.5)        # 更看重终端
            patch["_hits"].append("smoother")

        # D.“更保守/更安全”——扩大障碍半径、缩小转角上限
        if re.search(r"(更保守|更安全|safer|conservative)", t):
            patch["obstacle.radius"] = ("add", 0.4)         # +0.4m，显著绕远
            patch["constraints.delta_max"] = ("mul_clip", 0.7, 0.1, 0.6)
            patch["constraints.delta_min"] = ("sym_from", "constraints.delta_max")
            patch["_hits"].append("safer")

        # E.“安全距离 X m”
        m = re.search(r"(安全距离|safe distance).*?([0-9.]+)\s*(m|米)", t)
        if m:
            extra = float(m.group(2))
            patch["obstacle.radius"] = ("add", extra)       # 再叠加
            patch["_hits"].append("safe_distance")

        # F.“转向角限制 X deg/rad”
        m2 = re.search(r"(转向角|steer|steering).*?([0-9.]+)\s*(deg|degree|degrees|度|rad|弧度)", t)
        if m2:
            val = float(m2.group(2)); unit = m2.group(3)
            rad = _to_rad(val, unit)
            patch["constraints.delta_max"] = ("set", float(rad))
            patch["constraints.delta_min"] = ("set", -float(rad))
            patch["_hits"].append("steer_limit")

        # G.“更远/更短的视野”
        if re.search(r"(更远|长视野|longer)", t):
            patch["horizon"] = ("add_int", +20)
            patch["_hits"].append("longer")
        if re.search(r"(更短|shorter)", t):
            patch["horizon"] = ("add_int", -20)
            patch["_hits"].append("shorter")

        # H. 风险等级上下文
        risk = (context or {}).get("risk_hint")
        if risk in ["low", "med", "high"]:
            patch["__risk_level__"] = ("set", risk)
            if risk == "high":
                patch["obstacle.radius"] = ("add", 0.3)     # 高风险再增加安全裕度
                patch["weights.control"] = ("mul", 1.5)
                patch["u_rate_weight"] = ("set", 0.5)
            patch["_hits"].append(f"risk:{risk}")

        return patch

    def parse(self, text, context=None):
        context = context or {}
        if self.provider is None:
            return self._fallback_rules(text, context)

        prompt = f"""
You are a control-semantic planner. Turn instruction+context into a JSON patch for an MPC DSL.
Allowed operations:
  ("set", v) | ("add", x) | ("mul", r) | ("add_int", k) | ("scale", r)
  ("mul_clip", r, lo, hi) | ("sym_from", "key.path") | ("vec2", a, b)
Keys you may output:
  horizon, dt, terminal_scale, weights.control, obstacle.radius,
  constraints.delta_max, constraints.delta_min, u_rate_weight,
  goal_bias (2D), bias.side, __risk_level__
Return JSON only.
INSTRUCTION: {text}
CONTEXT: {json.dumps(context, ensure_ascii=False)}
"""
        try:
            raw = self.provider(prompt)
            return json.loads(raw)
        except Exception:
            # provider 输出非 JSON 或异常，回退到强规则
            print("[LLMInterpreter] provider returned non-JSON, fallback to rules.")
            return self._fallback_rules(text, context)
