"""Headless Blender process boundary tests, without an installed Blender binary."""
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

from PIL import Image
import pytest

from img2three import blender


@pytest.mark.parametrize("options", [
    {"engine": "UNKNOWN"}, {"resolution": 0}, {"resolution": -2}, {"resolution": 2.5},
    {"samples": 0}, {"samples": -1}, {"samples": 1.5},
    {"timeout": 0}, {"timeout": -1}, {"timeout": float("nan")}, {"timeout": float("inf")},
])
def test_invalid_render_options(options):
    with pytest.raises(ValueError):
        blender.BlenderOptions(**options)


@pytest.fixture
def fake_executable(tmp_path):
    executable = tmp_path / "Blender install" / "blender.exe"
    executable.parent.mkdir()
    executable.touch()
    return executable


def test_explicit_blender_path_wins_over_environment(monkeypatch, fake_executable):
    monkeypatch.setenv("BLENDER_PATH", "missing-env-blender")
    assert Path(blender.find_blender(str(fake_executable))) == fake_executable


def test_blender_environment_path(monkeypatch, fake_executable):
    monkeypatch.setenv("BLENDER_PATH", str(fake_executable))
    assert Path(blender.find_blender()) == fake_executable


def test_blender_found_on_path(monkeypatch, fake_executable):
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    monkeypatch.setattr(blender.shutil, "which", lambda _name: str(fake_executable))
    assert Path(blender.find_blender()) == fake_executable


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_invalid_override_does_not_silently_select_another_blender(monkeypatch, fake_executable, source):
    missing = str(fake_executable.parent / "missing.exe")
    # A valid PATH candidate must not override an explicitly requested installation.
    monkeypatch.setattr(blender.shutil, "which", lambda name: str(fake_executable) if name == "blender" else None)
    monkeypatch.setenv("BLENDER_PATH", missing if source == "environment" else str(fake_executable))
    with pytest.raises(FileNotFoundError):
        blender.find_blender(missing if source == "explicit" else None)


@pytest.fixture
def mesh_path(tmp_path):
    mesh = tmp_path / "mesh with spaces.glb"
    mesh.write_bytes(b"test-mesh")
    return mesh


def _job_from_command(command):
    assert isinstance(command, list)
    assert command[command.index("--python") + 1].endswith("_blender_scene.py")
    assert command[-2] == "--"
    return json.loads(Path(command[-1]).read_text(encoding="utf-8"))


def _write_render_outputs(job, *, include_blend=True):
    render_path = Path(job["render_path"])
    render_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), (100, 150, 200)).save(render_path)
    if include_blend:
        Path(job["blend_path"]).write_bytes(b"BLENDER-test-scene")
    if job["frames_dir"]:
        frames_dir = Path(job["frames_dir"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(job["n_views"]):
            Image.new("RGB", (16, 16), (20 * i, 40, 80)).save(frames_dir / f"frame_{i:04d}.png")


def test_render_invocation_and_artifacts(tmp_path, monkeypatch, mesh_path, fake_executable):
    out_dir = tmp_path / "final output"
    seen = {}

    def run(command, **kwargs):
        assert command[0] == str(fake_executable)
        assert "--background" in command and "--factory-startup" in command
        assert command[command.index("--python-exit-code") + 1] == "1"
        assert kwargs.get("shell", False) is False
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 45
        assert kwargs["stderr"] == subprocess.STDOUT
        assert hasattr(kwargs["stdout"], "write")
        job = _job_from_command(command)
        assert Path(job["mesh_path"]) == mesh_path.resolve()
        assert Path(job["render_path"]).parent != out_dir
        assert job["mode"] == "depth"
        assert job["engine"] == "BLENDER_EEVEE_NEXT"
        assert job["resolution"] == 96 and job["samples"] == 4
        assert job["frames_dir"] is None
        assert job["fov"] == 65.0 and job["aspect_ratio"] == 2.0
        seen["job_file"] = Path(command[-1])
        _write_render_outputs(job)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(blender.subprocess, "run", run)
    opts = blender.BlenderOptions(executable=str(fake_executable), engine="BLENDER_EEVEE_NEXT",
                                  resolution=96, samples=4, timeout=45)
    files = blender.render_mesh(mesh_path, out_dir, opts, prefix="depth", mode="depth", fov=65.0, aspect_ratio=2.0)

    assert {"render", "blend"} <= files.keys()
    assert "turntable" not in files and "views" not in files
    assert all(isinstance(files[key], Path) and files[key].is_file() for key in ("render", "blend"))
    assert files["render"].parent == out_dir
    assert files["blend"].read_bytes() == b"BLENDER-test-scene"
    with Image.open(files["render"]) as image:
        assert image.size == (16, 8)
    assert not seen["job_file"].exists()


def test_turntable_creates_animation_and_contact_sheet(tmp_path, monkeypatch, mesh_path, fake_executable):
    def run(command, **_kwargs):
        job = _job_from_command(command)
        assert job["frames_dir"] is not None
        assert job["n_views"] == 6
        _write_render_outputs(job)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(blender.subprocess, "run", run)
    files = blender.render_mesh(mesh_path, tmp_path / "out", blender.BlenderOptions(executable=str(fake_executable)),
                                turntable=True, n_views=6)

    assert {"render", "blend", "turntable", "views"} <= files.keys()
    with Image.open(files["turntable"]) as animation:
        assert animation.n_frames == 6
    with Image.open(files["views"]) as sheet:
        assert sheet.width > sheet.height


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_process_failure_includes_blender_diagnostic(tmp_path, monkeypatch, mesh_path, fake_executable, failure):
    def run(command, **kwargs):
        message = "Blender diagnostic: render device unavailable\n"
        try:
            kwargs["stdout"].write(message)
        except TypeError:
            kwargs["stdout"].write(message.encode("utf-8"))
        kwargs["stdout"].flush()
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(blender.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="render device unavailable"):
        blender.render_mesh(mesh_path, tmp_path / "out", blender.BlenderOptions(executable=str(fake_executable)))


def test_stale_outputs_do_not_hide_incomplete_render(tmp_path, monkeypatch, mesh_path, fake_executable):
    out_dir = tmp_path / "out"

    def successful_run(command, **_kwargs):
        _write_render_outputs(_job_from_command(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(blender.subprocess, "run", successful_run)
    opts = blender.BlenderOptions(executable=str(fake_executable))
    files = blender.render_mesh(mesh_path, out_dir, opts)
    files["render"].write_bytes(b"previous-render")
    files["blend"].write_bytes(b"previous-scene")

    def incomplete_run(command, **_kwargs):
        _write_render_outputs(_job_from_command(command), include_blend=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(blender.subprocess, "run", incomplete_run)
    with pytest.raises(RuntimeError):
        blender.render_mesh(mesh_path, out_dir, opts)
    assert files["render"].read_bytes() == b"previous-render"
    assert files["blend"].read_bytes() == b"previous-scene"


def test_missing_mesh_does_not_start_blender(tmp_path, monkeypatch, fake_executable):
    run = Mock(side_effect=AssertionError("Blender must not run without input"))
    monkeypatch.setattr(blender.subprocess, "run", run)
    with pytest.raises(FileNotFoundError):
        blender.render_mesh(tmp_path / "missing.glb", tmp_path / "out",
                            blender.BlenderOptions(executable=str(fake_executable)))
    run.assert_not_called()
