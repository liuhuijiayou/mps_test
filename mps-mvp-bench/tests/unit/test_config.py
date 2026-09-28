"""Config validation, override precedence, unknown-key rejection."""

import os
import textwrap

import pytest

from mps_bench import config as cfgmod

CASES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "cases")


def test_defaults_validate():
    cfg = cfgmod.load()
    assert cfg["measure.profile"] == "standard"
    assert cfg["gpu.uuid"] is None
    assert cfg["docker.runtime_args"] == []


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("measure:\n  duration_sec: 10\n", encoding="utf-8")
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.load(config_files=[str(path)])
    assert "未知配置项" in str(exc.value)


def test_unknown_set_key_rejected():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["measure.nope=1"])


def test_range_validation():
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.load(overrides=["clients.a.mps.active_thread_percentage=150"])
    assert "最大值" in str(exc.value)
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["measure.repeats=0"])


def test_choice_validation():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["clients.a.model.precision=int4"])


def test_type_validation_rejects_bool_for_int():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["measure.repeats=true"])


def test_override_precedence(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("measure:\n  duration_s: 60\n  repeats: 9\n", encoding="utf-8")
    case = tmp_path / "case.yaml"
    case.write_text(textwrap.dedent("""
        case:
          id: T1
          mode: nonmps_concurrent
        measure:
          duration_s: 30
    """), encoding="utf-8")
    cfg = cfgmod.load(config_files=[str(base)], case_file=str(case),
                      overrides=["measure.duration_s=11"])
    # cli --set beats case beats runtime yaml beats defaults
    assert cfg["measure.duration_s"] == 11
    assert cfg["measure.repeats"] == 9
    assert cfg["case.id"] == "T1"


def test_profile_presets():
    smoke = cfgmod.load(profile="smoke")
    standard = cfgmod.load(profile="standard")
    assert smoke["measure.duration_s"] == 15 and smoke["measure.repeats"] == 1
    assert standard["measure.duration_s"] == 120 and standard["measure.repeats"] == 5
    assert standard["telemetry.require_sm_metrics"] is True


def test_nonmps_cannot_declare_mps_quota():
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.load(overrides=["case.mode=nonmps_concurrent",
                               "clients.a.mps.active_thread_percentage=50"])
    assert "不能伪装存在 MPS 配额" in str(exc.value)


def test_solo_requires_solo_client():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["case.mode=mps_solo"])
    cfg = cfgmod.load(overrides=["case.mode=mps_solo", "case.solo_client=a"])
    assert cfg["case.solo_client"] == "a"


def test_online_requires_open_loop_and_qps():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["clients.a.mode=online",
                               "clients.a.load.arrival=closed_loop"])
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["clients.a.mode=online", "clients.a.load.target_qps=null"])


def test_offline_requires_closed_loop():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["clients.a.mode=offline", "clients.a.load.arrival=poisson"])


def test_static_partition_requires_capability_flag_and_excludes_atp():
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.load(overrides=["clients.a.mps.static_partition=partA"])
    assert "static_partitioning.enabled" in str(exc.value)

    parts = '[{"name": "partA", "sm_count": 28}]'
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.load(overrides=["mps.static_partitioning.enabled=true",
                               f"mps.static_partitioning.partitions={parts}",
                               "clients.a.mps.static_partition=partA",
                               "clients.a.mps.active_thread_percentage=50"])
    assert "分开配置" in str(exc.value)

    cfg = cfgmod.load(overrides=["mps.static_partitioning.enabled=true",
                                 f"mps.static_partitioning.partitions={parts}",
                                 "clients.a.mps.static_partition=partA"])
    assert cfg["clients.a.mps.static_partition"] == "partA"


def test_disruptive_fault_requires_flag():
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.load(overrides=["faults.enabled=true", "case.fault=F6"])
    assert "破坏性用例" in str(exc.value)


def test_docker_socket_mount_always_rejected():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["docker.mount_docker_socket=true"])


def test_require_sm_metrics_conflicts_with_degraded():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["telemetry.require_sm_metrics=true",
                               "telemetry.allow_degraded=true"])


def test_sm_active_field_mandatory_when_required():
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.load(overrides=["telemetry.dcgm_fields=[1003,1005]"])
    assert "1002" in str(exc.value)


def test_duplicate_ports_rejected():
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.load(overrides=["clients.b.load.port=18801"])


def test_hash_and_dump_roundtrip():
    cfg = cfgmod.load()
    other = cfgmod.load()
    assert cfg.hash() == other.hash()
    changed = cfgmod.load(overrides=["measure.seed=1"])
    assert changed.hash() != cfg.hash()
    assert "measure" in cfg.dump_yaml()


def test_schema_documents_every_field():
    rows = cfgmod.describe_schema()
    assert len(rows) == len(cfgmod.SCHEMA)
    for row in rows:
        assert row["type"]
        assert row["stage"]


def test_all_shipped_cases_validate():
    """Every committed case file must load and validate."""
    import glob
    files = glob.glob(os.path.join(CASES_DIR, "**", "*.yaml"), recursive=True)
    assert files, "未找到用例文件"
    for path in files:
        cfgmod.load(case_file=path)
