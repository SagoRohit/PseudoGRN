"""Compatibility shim restoring the `EstimateMI` import name.

WHY THIS FILE EXISTS (diagnosed, not guessed -- verified against this repo's
own git history before writing this file)
--------------------------------------------------------------------------
`main.py` (and `run_4new pseudo_methods_example.py`) do `from EstimateMI
import *`, but no `EstimateMI.py` ships in the repo as cloned -- that import
fails immediately.

`git log --all --follow -- psedoScore.py` shows the missing file was never
lost: commit `2d1188c` ("Rename EstimateMI.py to psedoScore.py") renamed it
in place. Every symbol `main.py` needs (`cal_mi2`, the Mixed-KSG mutual-
information estimator used for STEP2 of the pipeline) still exists,
unchanged in behavior, in `psedoScore.py` -- the rename just never got
propagated to the two files that imported it by the old name.

This file is a pure re-export -- it adds no logic, changes no scoring
behavior, and is not a reimplementation. It exists only so the two
call sites that still say `from EstimateMI import *` keep working, without
our having to edit `main.py` itself (kept byte-for-byte as cloned).

NOT fixed by this shim: `run_slingshot_distance_example.py` additionally
imports `cal_pearson2`, `cal_symmetric_pearson2`, `cal_js_pearson2`, and
`cal_neyman2` from `psedoScore`, none of which exist anywhere in this
repo's history (confirmed via `git log --all -- '*EstimateMI*' '*psedoScore*'`
and a symbol search across every commit) -- these are a genuinely missing,
un-recovered part of the authors' own ablation code, not a renamed file.
Our harness does not depend on that script or those symbols: our
train_sergio.py drives the same STEP1 (smooth) -> STEP2 (cal_mi2, Mixed-KSG
mutual information) -> STEP3 (MRMR2, redundancy-penalized greedy
re-ranking) pipeline `main.py` itself uses, which this shim fully restores.
"""
from psedoScore import *  # noqa: F401,F403
