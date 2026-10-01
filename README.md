# Atomic Electronic-Structure Calculations

Originally developed in MATLAB as a computational physics coursework project. This repository also contains a Python port under development. **The Python solver has not been validated against MATLAB; neither uploaded example is currently presented as a verified reference calculation.**

The two languages implement the same project, not two independent research projects. The project explores self-consistent atomic electronic-structure calculations using B-splines and Gaussian quadrature. The precise exchange approximation and total-energy expression still need to be checked against the original coursework description before making quantitative accuracy claims.

## Repository layout

| Directory | Purpose | Status |
| --- | --- | --- |
| `matlab/` | Original uploaded coursework implementation | Preserved unchanged; example configuration needs repair |
| `python/` | Python functions and demonstration notebook | Development draft; known numerical/runtime blockers |
| `validation/` | Repair sequence and comparison criteria | No validated reference results yet |
| `archive/` | Original README and notebook | Historical snapshot, not the active entry point |

The active notebook imports `python/modules.py` instead of maintaining duplicate function definitions. Its historical calculation example is disabled pending repair. The original notebook is retained for comparison, including its saved error output.

## Inspect the Python draft

From the repository root, create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab python/periodic_table.ipynb
```

These commands open the development notebook; they do not establish that the solver works. The dependency list is an initial list, not a numerically validated or locked environment. See [validation plan](validation/PLAN.md) before enabling calculations.

## Contributions and provenance

The MATLAB work originated in a university coursework project. The repository owner reports that the Python translation may have been AI-assisted; its exact provenance and completeness are not yet confirmed. Publication of a Python file is not evidence that its output is correct.

This local reorganisation preserves the MATLAB and Python source files byte-for-byte. It changes documentation and the active notebook structure only. No new software licence is assigned: author contributions and any reused course code should be established first. This repository is separate from the owner's master's thesis code.

## Next milestone

Reproduce one verified MATLAB atomic calculation, repair the Python numerical primitives, and compare both implementations with identical parameters. Only after that should this project claim a validated Python implementation. Planned tests cover B-spline behaviour, quadrature, generalised eigenproblem residuals, density normalisation, and self-consistent convergence.

## 中文说明

这是同一个课程项目的 MATLAB 原实现和 Python 移植草稿。Python 尚未完成结果验证；MATLAB 当前上传的示例也需要先修复参数问题。下一步先建立一个可信的参考算例，再完成 Python 修复和对比，而不是把能导入模块当作计算成功。
