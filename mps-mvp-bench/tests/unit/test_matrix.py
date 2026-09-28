"""Matrix expansion, budget gate and round interleaving."""

import os

import pytest

from mps_bench import config as cfgmod
from mps_bench import matrix as matrixmod

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASES_DIR = os.path.join(ROOT, "cases")


def test_discover_by_family():
    files = matrixmod.discover_case_files(CASES_DIR, families=["baselines"])
    assert files and all("baselines" in f for f in files)


def test_discover_by_id():
    files = matrixmod.discover_case_files(CASES_DIR, ids=["HH70"])
    assert files and len(files) == 3  # image-image / recsys-recsys / online-offline


def test_expand_includes_sweep_variants():
    files = matrixmod.discover_case_files(CASES_DIR, ids=["HL-C"])
    cases = matrixmod.expand(files)
    names = {c.variant for c in cases}
    assert "base" in names and "low70" in names
    low70 = next(c for c in cases if c.variant == "low70")
    assert low70.config["clients.b.mps.active_thread_percentage"] == 70
    assert low70.full_id == "HL-C[low70]"


def test_expand_skips_disabled_cases():
    files = matrixmod.discover_case_files(CASES_DIR, ids=["F6"])
    assert files, "F6 用例文件应存在"
    assert matrixmod.expand(files) == []  # disruptive -> enabled: false


def test_estimate_and_budget():
    files = matrixmod.discover_case_files(CASES_DIR, ids=["B0"])
    cases = matrixmod.expand(files, profile="smoke")
    est = matrixmod.estimate_duration_s(cases)
    assert est["case_count"] == 2
    assert est["estimated_total_s"] > 0
    assert matrixmod.check_budget(est, budget_s=10 ** 9) is None
    msg = matrixmod.check_budget(est, budget_s=1)
    assert msg and "超过" in msg


def test_sweep_must_have_set(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("case:\n  id: X\n  mode: nonmps_concurrent\n  sweep:\n    - name: a\n",
                    encoding="utf-8")
    with pytest.raises(cfgmod.ConfigError):
        matrixmod.expand([str(path)])


def test_interleave_abba():
    plan = matrixmod.interleave_order(["B2", "M0"], "ABBA", repeats=2)
    assert plan == [("B2", 0), ("M0", 0), ("M0", 1), ("B2", 1)]


def test_interleave_random_is_seeded():
    a = matrixmod.interleave_order(["x", "y", "z"], "random", repeats=3, seed=5)
    b = matrixmod.interleave_order(["x", "y", "z"], "random", repeats=3, seed=5)
    assert a == b


def test_interleave_unknown_mode():
    with pytest.raises(ValueError):
        matrixmod.interleave_order(["a"], "zigzag", repeats=1)
