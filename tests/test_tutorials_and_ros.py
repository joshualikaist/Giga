"""튜토리얼 스모크 테스트 + ROS2 호환성 테스트.

- 튜토리얼이 (헤드리스 모드로) 에러 없이 끝나는지 확인 → 코드를 고치다 튜토리얼이 망가지는 것 방지
- simple_biped.urdf를 ROS2의 robot_state_publisher가 실제로 읽을 수 있는지 확인
  (ROS2 환경이 로드되지 않았으면 건너뜀)
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from biped_sim import paths

TUTORIALS = paths.REPO_ROOT / "tutorials"


@pytest.mark.parametrize("script, args, must_contain", [
    ("t01_hello_mujoco.py", ["--headless", "--duration", "0.5"], "관찰 포인트"),
    ("t02_model_and_data.py", [], "수직력의 합"),
    ("t03_urdf_pendulum.py", ["--headless"], "[빌더 사용]"),
    ("t04_biped_inspect.py", ["--headless"], "MJCF 내보내기"),
    ("t05_biped_hanging_pd.py", ["--headless", "--duration", "2"], "추종 성능"),
    ("t06_biped_stand.py", ["--headless", "--duration", "2"], "서 있음 ✅"),
], ids=["t01", "t02", "t03", "t04", "t05", "t06"])
def test_tutorial_runs_headless(script, args, must_contain):
    proc = subprocess.run([sys.executable, str(TUTORIALS / script), *args],
                          capture_output=True, text=True, timeout=300, cwd=paths.REPO_ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert must_contain in proc.stdout, proc.stdout[-2000:]
    if script.startswith("t03"):
        # t03은 일부러 "URDF 그대로" 모델이 FAIL하는 것을 보여 주고, 빌더 모델은 PASS해야 함
        assert proc.stdout.split("[빌더 사용]", 1)[1].split("\n", 2)[1].rstrip().endswith("PASS ✅")


RSP = Path("/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher")


@pytest.mark.skipif(not os.environ.get("ROS_DISTRO") or not RSP.exists(),
                    reason="ROS2 Humble 환경이 로드되지 않음 (source scripts/activate.sh)")
def test_urdf_loads_in_ros2_robot_state_publisher(tmp_path):
    yaml = pytest.importorskip("yaml")
    params = tmp_path / "rsp.yaml"
    params.write_text(yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {
        "robot_description": paths.SIMPLE_BIPED_URDF.read_text()}}}, allow_unicode=True))

    # 노드는 스스로 종료하지 않으므로 3초 실행 후 SIGTERM으로 정상 종료시키고 출력 전체를 수집
    # 사용자가 실행 중인 노드(기본 ROS_DOMAIN_ID=27)와 섞이지 않도록 별도 번호에서 실행
    proc = subprocess.Popen([str(RSP), "--ros-args", "--params-file", str(params)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, ROS_DOMAIN_ID="86"))
    try:
        output, _ = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        proc.terminate()
        output, _ = proc.communicate(timeout=10)
    segments = {line.rsplit("got segment", 1)[1].strip()
                for line in output.splitlines() if "got segment" in line}
    expected = {"base_link", "left_thigh", "left_shin", "left_foot",
                "right_thigh", "right_shin", "right_foot"}
    assert segments == expected, output
