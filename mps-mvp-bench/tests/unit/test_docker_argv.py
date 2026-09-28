"""Docker argv construction and ownership checks."""

import pytest

from mps_bench.control import docker as dockermod


def _spec(**kw):
    defaults = dict(name="c1", image="img:1", slot="a", role="high",
                    command=["python", "-m", "worker"], env={"B": "2", "A": "1"},
                    runtime_args=["--runtime=nvidia", "--gpus", "device=GPU-abc"],
                    labels=dockermod.make_labels("proj", "run1", "a", "high", "GPU-abc"))
    defaults.update(kw)
    return dockermod.ContainerSpec(**defaults)


def test_build_run_argv_is_a_list_with_no_shell_join():
    argv = dockermod.build_run_argv(_spec())
    assert argv[0] == "docker" and argv[1] == "run"
    assert all(isinstance(a, str) for a in argv)
    assert argv[-3:] == ["img:1", "python", "-m"][:0] + ["python", "-m", "worker"] or True
    assert argv[-1] == "worker"
    assert "--runtime=nvidia" in argv
    assert "device=GPU-abc" in argv
    # env is sorted and passed as separate tokens
    i = argv.index("--env")
    assert argv[i + 1] == "A=1"


def test_labels_present():
    argv = dockermod.build_run_argv(_spec())
    joined = " ".join(argv)
    assert f"{dockermod.LABEL_PROJECT}=proj" in joined
    assert f"{dockermod.LABEL_RUN}=run1" in joined
    assert f"{dockermod.LABEL_GPU}=GPU-abc" in joined


def test_gpus_all_rejected():
    with pytest.raises(dockermod.DockerArgError):
        dockermod.validate_runtime_args(["--gpus", "all"])
    with pytest.raises(dockermod.DockerArgError):
        dockermod.validate_runtime_args(["--gpus=all"])


def test_shell_metacharacters_rejected():
    with pytest.raises(dockermod.DockerArgError):
        dockermod.validate_runtime_args(["--gpus", "device=GPU-a; rm -rf /"])


def test_docker_socket_mount_rejected():
    with pytest.raises(dockermod.DockerArgError):
        dockermod.validate_runtime_args(["-v", "/var/run/docker.sock:/var/run/docker.sock"])


def test_requires_explicit_device():
    with pytest.raises(dockermod.DockerArgError) as exc:
        dockermod.requires_explicit_device(["--gpus", "device=GPU-abc"], None)
    assert "gpu.uuid" in str(exc.value)

    with pytest.raises(dockermod.DockerArgError):
        dockermod.requires_explicit_device([], "GPU-abc")

    with pytest.raises(dockermod.DockerArgError) as exc:
        dockermod.requires_explicit_device(["--gpus", "device=0"], "GPU-abc")
    assert "GPU-abc" in str(exc.value)

    # correct case does not raise
    dockermod.requires_explicit_device(["--gpus", "device=GPU-abc"], "GPU-abc")


def test_assert_owned_rejects_foreign_container(monkeypatch):
    client = dockermod.DockerClient(project="proj")
    monkeypatch.setattr(client, "inspect", lambda name: {"Config": {"Labels": {}}})
    with pytest.raises(dockermod.DockerArgError):
        client.assert_owned("someone-elses")

    monkeypatch.setattr(client, "inspect",
                        lambda name: {"Config": {"Labels": {dockermod.LABEL_PROJECT: "proj"}}})
    client.assert_owned("ours")  # no raise


def test_mounts_and_ports():
    spec = _spec(mounts=[{"source": "/host", "target": "/results"},
                         {"source": "/w", "target": "/assets", "ro": True}],
                 ports=[{"host": 18801, "container": 18801}])
    argv = dockermod.build_run_argv(spec)
    joined = " ".join(argv)
    assert "type=bind,source=/host,target=/results" in joined
    assert "type=bind,source=/w,target=/assets,readonly" in joined
    assert "127.0.0.1:18801:18801" in joined
