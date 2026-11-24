"""
손글씨 숫자 인식의 핵심 기능을 모아 둔 파일입니다.
데스크탑 프로그램(app.py)과 웹 프로그램(web.py)이 이 파일을 함께 씁니다.
그래서 인식 방법을 고치면 두 프로그램에 똑같이 적용됩니다.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from model import DigitCNN

# 이 파일이 있는 폴더 기준으로 모델 파일 경로를 잡습니다.
현재_폴더 = os.path.dirname(os.path.abspath(__file__))
모델_경로 = os.path.join(현재_폴더, "mnist_cnn.pt")

캔버스_크기 = 280   # 그림판 크기 (28x28의 10배)
펜_굵기 = 22        # 글씨 굵기 (MNIST 글씨 두께와 비슷하게)


def 모델_불러오기():
    """저장된 학습 결과(가중치)를 읽어서 모델을 준비합니다."""
    if not os.path.exists(모델_경로):
        raise SystemExit("mnist_cnn.pt 파일이 없습니다. 먼저 train.py를 실행해 주세요.")
    모델 = DigitCNN()
    모델.load_state_dict(torch.load(모델_경로, map_location="cpu"))
    모델.eval()  # 평가 모드 (드롭아웃 끄기)
    return 모델


def 빈_그림_만들기():
    """인식에 쓸 검은 바탕 그림(280x280)을 만듭니다."""
    return Image.new("L", (캔버스_크기, 캔버스_크기), 0)


def 점_찍기(붓, x, y):
    """선 끝을 둥글게 만들기 위해 동그라미를 찍습니다."""
    r = 펜_굵기 / 2
    붓.ellipse([x - r, y - r, x + r, y + r], fill=255)


def 선_긋기(붓, 이전_점, 현재_점):
    """두 점 사이에 흰 선을 긋고, 끝에 동그라미를 찍습니다."""
    붓.line([이전_점, 현재_점], fill=255, width=펜_굵기)
    점_찍기(붓, *현재_점)


def 획으로_그림_만들기(획_목록):
    """
    웹 화면에서 보낸 획(점 좌표 목록들)을 데스크탑 프로그램과 똑같은 방법으로 다시 그립니다.
    획_목록 예: [[(10, 20), (12, 25), ...], [...]]  (획 하나 = 마우스를 누른 채 움직인 점들)
    """
    그림 = 빈_그림_만들기()
    붓 = ImageDraw.Draw(그림)
    for 획 in 획_목록:
        점들 = [(float(x), float(y)) for x, y in 획]
        if not 점들:
            continue
        점_찍기(붓, *점들[0])
        for 이전_점, 현재_점 in zip(점들, 점들[1:]):
            선_긋기(붓, 이전_점, 현재_점)
    return 그림


def mnist_형식으로_변환(그림):
    """
    사용자가 그린 큰 그림을 MNIST와 같은 형식(28x28, 가운데 정렬)으로 바꿉니다.
    MNIST 숫자는 20x20 칸 안에 들어가고, 무게중심이 28x28 한가운데에 오도록 만들어져 있어서
    똑같이 맞춰 줘야 인식이 잘 됩니다.
    """
    배열 = np.array(그림)
    글씨_위치 = np.argwhere(배열 > 0)
    if 글씨_위치.size == 0:
        return None  # 아무것도 안 그렸으면 인식하지 않습니다.

    # 1) 글씨가 있는 부분만 잘라냅니다.
    (위, 왼쪽), (아래, 오른쪽) = 글씨_위치.min(0), 글씨_위치.max(0) + 1
    잘린_그림 = 그림.crop((왼쪽, 위, 오른쪽, 아래))

    # 2) 가로세로 비율을 유지하면서 긴 쪽을 20픽셀로 줄입니다.
    가로, 세로 = 잘린_그림.size
    배율 = 20.0 / max(가로, 세로)
    새_가로, 새_세로 = max(1, round(가로 * 배율)), max(1, round(세로 * 배율))
    작은_그림 = 잘린_그림.resize((새_가로, 새_세로), Image.LANCZOS)

    # 3) 28x28 검은 바탕 가운데에 붙입니다.
    결과 = Image.new("L", (28, 28), 0)
    결과.paste(작은_그림, ((28 - 새_가로) // 2, (28 - 새_세로) // 2))

    # 4) 글씨의 무게중심이 정확히 한가운데(14, 14)에 오도록 옮깁니다.
    배열 = np.array(결과, dtype=np.float32)
    세로_좌표, 가로_좌표 = np.indices(배열.shape)
    전체_밝기 = 배열.sum()
    중심_세로 = (세로_좌표 * 배열).sum() / 전체_밝기
    중심_가로 = (가로_좌표 * 배열).sum() / 전체_밝기
    이동_가로, 이동_세로 = round(14 - 중심_가로), round(14 - 중심_세로)
    결과 = 결과.transform(결과.size, Image.AFFINE, (1, 0, -이동_가로, 0, 1, -이동_세로))

    return 결과


def 숫자_인식하기(모델, 그림):
    """
    280x280 그림을 받아서 어떤 숫자인지 알아맞힙니다.
    돌려주는 값: (28x28로 바꾼 그림, 숫자 0~9 각각의 확률 배열)
    아무것도 안 그렸으면 (None, None)을 돌려줍니다.
    """
    작은_그림 = mnist_형식으로_변환(그림.filter(ImageFilter.GaussianBlur(1)))
    if 작은_그림 is None:
        return None, None

    # 0~1 사이 값으로 바꾸고 (1장, 채널 1, 28, 28) 모양으로 만듭니다.
    입력 = torch.tensor(np.array(작은_그림), dtype=torch.float32) / 255.0
    입력 = 입력.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        점수 = 모델(입력)
        확률 = torch.softmax(점수, dim=1)[0].numpy()  # 점수를 합이 1인 확률로 바꿉니다.

    return 작은_그림, 확률
