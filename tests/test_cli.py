"""설치된 CLI와 결과 무결성의 회귀 검사입니다."""
import json
import subprocess
import sys
import pytest
from parity import __version__
from parity.io import verify_bundle, new_dir
from parity.smoke import create_fixture


def test_package_works_outside_source_directory(tmp_path):
    result = subprocess.run([sys.executable, "-m", "parity", "--version"], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == __version__


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1"])
def test_invalid_tolerance_is_rejected(tmp_path, value):
    result = subprocess.run([sys.executable, "-m", "parity", "compare", "--left", "missing",
                             "--right", "missing", "--out", str(tmp_path / "report"), "--atol=" + value],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "허용치" in result.stderr


def test_undeclared_bundle_file_is_rejected(tmp_path):
    bundle = create_fixture(tmp_path / "bundle")
    (bundle / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(bundle)


def test_existing_output_is_not_overwritten(tmp_path):
    (tmp_path / "existing.txt").write_text("keep")
    with pytest.raises(ValueError):
        new_dir(tmp_path)
    assert (tmp_path / "existing.txt").read_text() == "keep"


def test_offline_smoke_cli(tmp_path):
    out = tmp_path / "smoke"
    subprocess.run([sys.executable, "-m", "parity", "smoke", "--out", str(out)], check=True,
                   capture_output=True, text=True)
    status = json.loads((out / "smoke.json").read_text())
    assert status["synthetic_weights"] is True
    assert status["completed"] is True
    assert (out / "report/report.md").is_file()
    assert (out / "matmul/reference_metrics.json").is_file()
