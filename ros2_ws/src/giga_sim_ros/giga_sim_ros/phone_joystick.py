"""phone_joystick — 핸드폰 브라우저를 조이스틱으로: 웹 페이지를 띄우고, 받은 스틱 값을 /cmd_vel로 (docs/11).

    ros2 run giga_sim_ros phone_joystick                 # 또는 duck.launch.py phone:=true
    → 같은 Wi-Fi의 핸드폰 브라우저에서 터미널에 나온 주소 (예: http://192.168.0.14:8080) — 앱 설치 필요 없음

화면: 왼쪽 스틱 = 앞뒤(위·아래)·옆(왼·오른), 오른쪽 스틱 = 회전, 가운데 [멈춤] [처음으로] 버튼과 속도 배율.
스틱을 놓으면 0. 핸드폰은 초당 약 15번 지금 스틱 값을 보내고, 이 노드는 그 값을 20 Hz로 /cmd_vel에 낸다.
안전장치: 0.5초 동안 핸드폰에서 아무것도 오지 않으면(화면 꺼짐, Wi-Fi 끊김) (0, 0, 0) = 멈춤.

파이썬 기본 라이브러리(http.server)만 씀 — 웹 서버를 따로 설치하지 않아도 됨.
같은 Wi-Fi인데 핸드폰에서 안 열리면 PC 방화벽: sudo ufw allow 8080/tcp
"""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from .shutdown import init_with_stop_flag

STALE_S = 0.5   # 이 시간 동안 핸드폰 소식이 없으면 멈춤


def lan_address() -> str:
    """이 PC의 같은 Wi-Fi 주소 (밖으로 나가는 경로의 주소. 실제로 패킷을 보내지는 않음)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 9))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class PhoneJoystick(Node):

    def __init__(self):
        super().__init__('phone_joystick')
        self.port = self.declare_parameter('port', 8080).value
        # 스틱을 끝까지 밀었을 때의 명령 (duck.launch.py가 정책의 학습 범위를 넘김)
        self.limits = {k: self.declare_parameter(k, v).value
                       for k, v in (('vx_max', 0.25), ('vx_min', -0.15), ('vy_max', 0.15), ('wz_max', 0.5))}
        self.lock = threading.Lock()
        self.stick = (0.0, 0.0, 0.0)          # 핸드폰이 보낸 명령 (이미 m/s, rad/s)
        self.last_heard = 0.0
        self.position = None
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.reset = self.create_client(Trigger, 'policy/reset')
        self.create_subscription(TFMessage, 'tf', self.on_tf, 10)
        self.create_timer(0.05, self.publish)

        node = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'      # 연결을 계속 써서 보내는 지연을 줄임

            def log_message(self, *args):      # 요청마다 찍지 않음
                pass

            def _send(self, code, body: bytes, ctype='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self._send(200, PAGE.replace('__LIMITS__', json.dumps(node.limits)).encode(), 'text/html; charset=utf-8')
                else:
                    self._send(404, b'{}')

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                try:
                    data = json.loads(self.rfile.read(length) or b'{}')
                except json.JSONDecodeError:
                    self._send(400, b'{}')
                    return
                if self.path == '/cmd':
                    self._send(200, json.dumps(node.on_phone(data)).encode())
                elif self.path == '/reset':
                    self._send(200, json.dumps({'ok': node.call_reset()}).encode())
                else:
                    self._send(404, b'{}')

        self.server = ThreadingHTTPServer(('0.0.0.0', self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        url = f'http://{lan_address()}:{self.port}'
        self.get_logger().info(f'핸드폰 조이스틱: 같은 Wi-Fi의 핸드폰 브라우저에서  {url}  (이 PC에서는 http://localhost:{self.port})')

    # -------------------------------------------------------------------- 핸드폰 → 명령
    def on_phone(self, data: dict) -> dict:
        """웹 페이지가 보낸 명령 (스틱 × 범위 × 배율을 페이지에서 계산해 m/s로 보냄) — 범위로 한 번 더 자름."""
        lim = self.limits

        def clip(v, lo, hi):
            try:
                return min(max(float(v), lo), hi)
            except (TypeError, ValueError):
                return 0.0
        cmd = (clip(data.get('vx'), lim['vx_min'], lim['vx_max']), clip(data.get('vy'), -lim['vy_max'], lim['vy_max']),
               clip(data.get('wz'), -lim['wz_max'], lim['wz_max']))
        with self.lock:
            self.stick, self.last_heard = cmd, time.monotonic()
            pos = self.position
        return {'cmd': cmd, 'position': pos}

    def call_reset(self) -> bool:
        if not self.reset.service_is_ready():
            return False
        self.reset.call_async(Trigger.Request())
        return True

    def on_tf(self, msg: TFMessage):
        for t in msg.transforms:
            if t.child_frame_id == 'base_link':
                p = t.transform.translation
                with self.lock:
                    self.position = (round(p.x, 2), round(p.y, 2))

    def publish(self):
        with self.lock:
            fresh = time.monotonic() - self.last_heard < STALE_S
            cmd = self.stick if fresh else (0.0, 0.0, 0.0)
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = cmd
        self.pub.publish(msg)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>오리 조이스틱</title>
<style>
  :root { --bg:#10151c; --panel:#1b2430; --line:#3a4a5e; --knob:#f2b84b; --text:#e8eef5; --dim:#8da0b5; --stop:#d9534f; }
  * { box-sizing:border-box; -webkit-user-select:none; user-select:none; -webkit-tap-highlight-color:transparent; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
               font-family:-apple-system, "Noto Sans KR", sans-serif; overflow:hidden; touch-action:none; }
  #app { display:grid; grid-template-columns:1fr auto 1fr; height:100%; align-items:center; gap:8px; padding:12px; }
  .stick { position:relative; width:min(42vw, 72vh); aspect-ratio:1; justify-self:center; border-radius:50%;
           background:radial-gradient(circle, var(--panel) 60%, #141b24); border:2px solid var(--line); touch-action:none; }
  .stick .knob { position:absolute; left:50%; top:50%; width:34%; height:34%; margin:-17% 0 0 -17%; border-radius:50%;
                 background:var(--knob); box-shadow:0 4px 14px #0008; pointer-events:none; }
  .stick .label { position:absolute; width:100%; text-align:center; bottom:-1.6em; color:var(--dim); font-size:14px; }
  .stick.turn::before { content:""; position:absolute; left:8%; right:8%; top:50%; height:2px; background:var(--line); }
  #middle { display:flex; flex-direction:column; gap:10px; align-items:stretch; min-width:150px; }
  button { font-size:18px; padding:14px 10px; border-radius:12px; border:0; color:white; background:#2f4b6e; }
  button.stop { background:var(--stop); font-size:22px; font-weight:bold; }
  #status { font-size:13px; color:var(--dim); text-align:center; line-height:1.5; min-height:4.5em; }
  #status b { color:var(--text); }
  label { font-size:13px; color:var(--dim); text-align:center; }
  input[type=range] { width:100%; }
  @media (orientation: portrait) {
    #app { grid-template-columns:minmax(0, 1fr); grid-template-rows:auto 1fr 1fr; }
    #middle { order:-1; flex-direction:row; flex-wrap:wrap; justify-content:center; min-width:0; }
    #middle button { flex:1; }
    #middle label, #status { flex-basis:100%; }
    .stick { width:min(70vw, 34vh); }
  }
</style></head><body>
<div id="app">
  <div class="stick" id="move"><div class="knob"></div><div class="label">앞뒤 · 옆</div></div>
  <div id="middle">
    <button class="stop" id="stop">멈춤</button>
    <button id="reset">처음으로</button>
    <label>속도 배율 <span id="scaleText">60%</span><input type="range" id="scale" min="20" max="100" value="60"></label>
    <div id="status">연결 중…</div>
  </div>
  <div class="stick turn" id="turn"><div class="knob"></div><div class="label">회전</div></div>
</div>
<script>
const LIM = __LIMITS__;
const sticks = {move: {x: 0, y: 0, id: null}, turn: {x: 0, y: 0, id: null}};
let scale = 0.6, lastOk = 0, sending = false;

function bind(name, axes) {
  const el = document.getElementById(name), knob = el.querySelector('.knob'), s = sticks[name];
  function update(e) {
    const r = el.getBoundingClientRect(), R = r.width / 2;
    let x = (e.clientX - r.left - R) / (R * 0.75), y = (e.clientY - r.top - R) / (R * 0.75);
    if (axes === 'x') y = 0;
    const n = Math.hypot(x, y); if (n > 1) { x /= n; y /= n; }
    s.x = x; s.y = -y;
    knob.style.transform = `translate(${x * R * 0.75}px, ${y * R * 0.75}px)`;
  }
  function release(e) {
    if (e.pointerId !== s.id) return;
    s.id = null; s.x = 0; s.y = 0; knob.style.transform = ''; send();
  }
  el.addEventListener('pointerdown', e => { s.id = e.pointerId; el.setPointerCapture(e.pointerId); update(e); send(); });
  el.addEventListener('pointermove', e => { if (e.pointerId === s.id) update(e); });
  el.addEventListener('pointerup', release);
  el.addEventListener('pointercancel', release);
}
bind('move', 'xy'); bind('turn', 'x');

function command() {   // 스틱 → m/s, rad/s (위 = 앞, 왼쪽 = 왼쪽 옆걸음, 회전 스틱 왼쪽 = 왼쪽으로 돌기)
  const m = sticks.move, t = sticks.turn;
  const vx = (m.y >= 0 ? m.y * LIM.vx_max : -m.y * LIM.vx_min) * scale;
  return {vx, vy: -m.x * LIM.vy_max * scale, wz: -t.x * LIM.wz_max * scale};
}
async function send() {
  if (sending) return; sending = true;
  const c = command();
  try {
    const r = await fetch('/cmd', {method: 'POST', body: JSON.stringify(c), headers: {'Content-Type': 'application/json'}});
    const j = await r.json(); lastOk = Date.now();
    const p = j.position ? `위치 x ${j.position[0].toFixed(2)} m, y ${j.position[1].toFixed(2)} m` : '오리 위치 모름 (시뮬레이터가 떠 있나요?)';
    document.getElementById('status').innerHTML =
      `<b>앞뒤 ${j.cmd[0].toFixed(2)}</b> · 옆 ${j.cmd[1].toFixed(2)} m/s · <b>회전 ${j.cmd[2].toFixed(2)}</b> rad/s<br>${p}`;
  } catch (e) {
    document.getElementById('status').textContent = '연결 끊김 — PC와 같은 Wi-Fi인지 확인';
  }
  sending = false;
}
setInterval(send, 66);   // 스틱을 안 움직여도 초당 약 15번 (PC는 0.5초 동안 소식이 없으면 멈춤)
setInterval(() => { if (Date.now() - lastOk > 1500) document.getElementById('status').textContent = '연결 끊김 — PC와 같은 Wi-Fi인지 확인'; }, 500);

document.getElementById('stop').addEventListener('pointerdown', () => {
  for (const s of Object.values(sticks)) { s.x = 0; s.y = 0; s.id = null; }
  document.querySelectorAll('.knob').forEach(k => k.style.transform = ''); send();
});
document.getElementById('reset').addEventListener('click', () => fetch('/reset', {method: 'POST', body: '{}'}));
const sl = document.getElementById('scale');
sl.addEventListener('input', () => { scale = sl.value / 100; document.getElementById('scaleText').textContent = sl.value + '%'; });
</script></body></html>
"""


def main(args=None):
    stop = init_with_stop_flag(args)
    node = PhoneJoystick()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while not stop.is_set():
            executor.spin_once(timeout_sec=0.05)
    finally:
        node.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
