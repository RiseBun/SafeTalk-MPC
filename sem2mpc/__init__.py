# sem2mpc/__init__.py
# -*- coding: utf-8 -*-
"""
把 sem2mpc 的子包同时暴露为顶层包名，兼容两种启动方式：
- 包方式（仓库根目录）：python -m sem2mpc.sim.sim_runner ...
- 脚本方式（进入 sem2mpc 目录）：python -m sim.sim_runner ...
"""

import sys as _sys

# 把子包导入，再把它们注册到 sys.modules 的“顶层别名”
from . import compiler as _compiler
from . import sim as _sim
from . import semantics as _semantics

_sys.modules.setdefault('compiler', _compiler)
_sys.modules.setdefault('sim', _sim)
_sys.modules.setdefault('semantics', _semantics)

# 同时把 providers 子包也挂一下（可选，但安全）
from .semantics import providers as _providers
_sys.modules.setdefault('providers', _providers)  # 仅当有相对导入 from providers... 时
