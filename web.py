r"""
손글씨 숫자 인식 프로그램의 웹 버전입니다.
이 파일을 실행하면 내 컴퓨터에 작은 웹 서버가 켜지고, 웹 브라우저에서 숫자를 쓸 수 있습니다.
추가로 설치할 것은 없습니다 (파이썬 기본 기능 http.server만 씁니다).

실행 방법:
    .venv\Scripts\python.exe web.py
    → 브라우저가 자동으로 열립니다. 안 열리면 http://127.0.0.1:8000 에 접속하세요.
    → 끝낼 때는 이 창에서 Ctrl+C를 누릅니다.

옵션:
    --port 8080       다른 포트 번호를 씁니다 (기본 8000).
    --host 0.0.0.0    같은 와이파이의 휴대폰·태블릿에서도 접속할 수 있게 합니다.
                      (기본값 127.0.0.1은 이 컴퓨터에서만 접속됩니다.)
    --no-browser      브라우저를 자동으로 열지 않습니다.

동작 방식:
    브라우저는 그린 획의 좌표만 서버로 보내고, 서버가 데스크탑 버전과 똑같은 방법
    (recognizer.py)으로 그림을 다시 그려서 인식한 뒤 결과를 돌려줍니다.
"""

import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from recognizer import 모델_불러오기, 숫자_인식하기, 획으로_그림_만들기

현재_폴더 = os.path.dirname(os.path.abspath(__file__))
화면_파일 = os.path.join(현재_폴더, "web", "index.html")

최대_요청_크기 = 1_000_000  # 너무 큰 요청은 받지 않습니다 (1MB)
모델 = None                  # 서버를 켤 때 한 번만 불러옵니다.
모델_잠금 = threading.Lock()  # 여러 요청이 동시에 와도 모델을 한 번에 하나씩 쓰도록 합니다.


class 요청_처리기(BaseHTTPRequestHandler):
    """브라우저의 요청을 받아서 화면(HTML)이나 인식 결과(JSON)를 돌려줍니다."""

    def do_GET(self):
        """주소창으로 들어오면 그림판 화면을 보내 줍니다."""
        if self.path not in ("/", "/index.html"):
            self.send_error(404, "Not Found")
            return
        with open(화면_파일, "rb") as 파일:
            내용 = 파일.read()
        self.응답_보내기(200, 내용, "text/html; charset=utf-8")

    def do_POST(self):
        """/predict 로 획 좌표를 받으면 숫자를 인식해서 결과를 보내 줍니다."""
        if self.path != "/predict":
            self.send_error(404, "Not Found")
            return
        try:
            길이 = int(self.headers.get("Content-Length", 0))
            if 길이 <= 0 or 길이 > 최대_요청_크기:
                raise ValueError("요청 크기가 올바르지 않습니다.")
            요청 = json.loads(self.rfile.read(길이).decode("utf-8"))
            그림 = 획으로_그림_만들기(요청["strokes"])
        except (ValueError, KeyError, TypeError) as 오류:
            self.json_보내기(400, {"error": f"잘못된 요청입니다: {오류}"})
            return

        with 모델_잠금:
            작은_그림, 확률 = 숫자_인식하기(모델, 그림)

        if 작은_그림 is None:
            self.json_보내기(200, {"digit": None})
            return

        숫자 = int(확률.argmax())
        self.json_보내기(200, {
            "digit": 숫자,
            "confidence": float(확률[숫자]),
            "probabilities": [float(값) for 값 in 확률],
            # 모델이 실제로 보는 28x28 이미지 (밝기 0~255, 784개)
            "preview": np.array(작은_그림).astype(int).flatten().tolist(),
        })

    def json_보내기(self, 상태, 데이터):
        self.응답_보내기(상태, json.dumps(데이터).encode("utf-8"), "application/json")

    def 응답_보내기(self, 상태, 내용, 형식):
        self.send_response(상태)
        self.send_header("Content-Type", 형식)
        self.send_header("Content-Length", str(len(내용)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(내용)

    def log_message(self, 형식, *값들):
        """요청마다 찍히는 기록은 생략합니다 (창이 지저분해지지 않게)."""
        pass


def 실행():
    global 모델
    설정 = argparse.ArgumentParser(description="손글씨 숫자 인식 웹 버전")
    설정.add_argument("--host", default="127.0.0.1", help="접속을 받을 주소 (기본 127.0.0.1)")
    설정.add_argument("--port", type=int, default=8000, help="포트 번호 (기본 8000)")
    설정.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않기")
    옵션 = 설정.parse_args()

    print("모델을 불러오는 중입니다...")
    모델 = 모델_불러오기()

    서버 = ThreadingHTTPServer((옵션.host, 옵션.port), 요청_처리기)
    주소 = f"http://127.0.0.1:{옵션.port}/"
    print(f"웹 버전이 켜졌습니다: {주소}")
    if 옵션.host == "0.0.0.0":
        print("같은 네트워크의 다른 기기에서는 http://<이 컴퓨터의 IP>:%d/ 로 접속하세요." % 옵션.port)
    print("끝내려면 Ctrl+C를 누르세요.")

    if not 옵션.no_browser:
        webbrowser.open(주소)
    try:
        서버.serve_forever()
    except KeyboardInterrupt:
        print("\n웹 버전을 끕니다.")
    finally:
        서버.server_close()


if __name__ == "__main__":
    실행()
