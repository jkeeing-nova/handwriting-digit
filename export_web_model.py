r"""
학습된 모델(mnist_cnn.pt)을 브라우저(JavaScript)가 읽을 수 있는 파일로 내보냅니다.
GitHub Pages 버전(docs/ 폴더)은 Python 없이 브라우저 안에서 인식하기 때문에 이 파일이 필요합니다.

실행 방법:
    .venv\Scripts\python.exe export_web_model.py

만들어지는 파일:
    docs/model.bin   가중치 숫자들 (float32, 리틀 엔디언으로 차례대로 이어 붙임)
    docs/model.json  각 층의 이름, 모양, model.bin 안에서의 시작 위치

train.py로 다시 학습했다면 이 스크립트도 다시 실행해야 웹(GitHub Pages) 버전에 반영됩니다.
"""

import json
import os

import numpy as np

from recognizer import 모델_불러오기, 캔버스_크기, 펜_굵기

현재_폴더 = os.path.dirname(os.path.abspath(__file__))
내보낼_폴더 = os.path.join(현재_폴더, "docs")


def 내보내기():
    모델 = 모델_불러오기()
    # model.py의 층 위치와 JavaScript에서 쓸 이름을 짝지어 둡니다.
    층_목록 = [
        ("conv1", 모델.특징추출[0]),   # 3x3 필터 32개
        ("conv2", 모델.특징추출[3]),   # 3x3 필터 64개
        ("fc1", 모델.분류[1]),        # 3136 -> 128
        ("fc2", 모델.분류[4]),        # 128 -> 10
    ]

    정보 = {"canvasSize": 캔버스_크기, "penWidth": 펜_굵기, "tensors": {}}
    조각들 = []
    위치 = 0  # model.bin 안에서 몇 번째 숫자부터인지
    for 이름, 층 in 층_목록:
        for 종류 in ("weight", "bias"):
            값 = getattr(층, 종류).detach().numpy().astype("<f4")
            정보["tensors"][f"{이름}.{종류}"] = {"shape": list(값.shape), "offset": 위치}
            조각들.append(값.ravel())
            위치 += 값.size

    os.makedirs(내보낼_폴더, exist_ok=True)
    np.concatenate(조각들).tofile(os.path.join(내보낼_폴더, "model.bin"))
    with open(os.path.join(내보낼_폴더, "model.json"), "w", encoding="utf-8") as 파일:
        json.dump(정보, 파일, indent=2)
    print(f"내보내기 완료: 숫자 {위치:,}개 ({위치 * 4 / 1024 / 1024:.2f} MB) → docs/model.bin, docs/model.json")


if __name__ == "__main__":
    내보내기()
