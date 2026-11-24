r"""
MNIST 손글씨 숫자 데이터로 CNN(합성곱 신경망) 모델을 학습하는 스크립트입니다.

실행 방법:
    .venv\Scripts\python.exe train.py

학습이 끝나면 같은 폴더에 mnist_cnn.pt(학습된 모델 파일)가 저장됩니다.
"""

import os
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DigitCNN

# 이 파일이 있는 폴더 기준으로 경로를 잡습니다.
현재_폴더 = os.path.dirname(os.path.abspath(__file__))
데이터_경로 = os.path.join(현재_폴더, "data", "mnist.npz")
모델_경로 = os.path.join(현재_폴더, "mnist_cnn.pt")

# MNIST 데이터 다운로드 주소 (Google이 제공하는 저장소, 약 11MB)
데이터_주소 = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def 데이터_불러오기():
    """MNIST 데이터가 없으면 내려받고, 학습용/테스트용 텐서로 바꿔서 돌려줍니다."""
    if not os.path.exists(데이터_경로):
        os.makedirs(os.path.dirname(데이터_경로), exist_ok=True)
        print("MNIST 데이터를 내려받는 중입니다...")
        urllib.request.urlretrieve(데이터_주소, 데이터_경로)

    with np.load(데이터_경로) as 파일:
        x_학습, y_학습 = 파일["x_train"], 파일["y_train"]
        x_테스트, y_테스트 = 파일["x_test"], 파일["y_test"]

    def 텐서로_변환(x, y):
        # 픽셀 값 0~255를 0~1 사이로 줄이고, (개수, 채널 1, 28, 28) 모양으로 바꿉니다.
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) / 255.0
        y = torch.tensor(y, dtype=torch.long)
        return TensorDataset(x, y)

    return 텐서로_변환(x_학습, y_학습), 텐서로_변환(x_테스트, y_테스트)


def 정확도_측정(모델, 로더):
    """테스트 데이터에서 모델이 몇 %를 맞히는지 계산합니다."""
    모델.eval()  # 평가 모드 (드롭아웃 끄기)
    맞은_개수 = 0
    전체_개수 = 0
    with torch.no_grad():  # 평가할 때는 기울기 계산이 필요 없습니다.
        for x, y in 로더:
            예측 = 모델(x).argmax(dim=1)  # 점수가 가장 높은 숫자를 고릅니다.
            맞은_개수 += (예측 == y).sum().item()
            전체_개수 += y.size(0)
    return 맞은_개수 / 전체_개수 * 100


def 이미지_흔들기(x):
    """학습 이미지를 조금씩 이동·확대·회전해서 다양한 글씨체에 강하게 만듭니다(데이터 증강)."""
    개수 = x.size(0)
    각도 = (torch.rand(개수) - 0.5) * 0.3          # 약 ±8.5도 회전
    배율 = 1 + (torch.rand(개수) - 0.5) * 0.2       # 약 ±10% 크기 변화
    이동 = (torch.rand(개수, 2) - 0.5) * 0.2        # 약 ±3픽셀 이동
    cos, sin = torch.cos(각도) / 배율, torch.sin(각도) / 배율
    변환행렬 = torch.stack(
        [torch.stack([cos, -sin, 이동[:, 0]], 1), torch.stack([sin, cos, 이동[:, 1]], 1)], 1
    )
    격자 = nn.functional.affine_grid(변환행렬, x.size(), align_corners=False)
    return nn.functional.grid_sample(x, 격자, align_corners=False)


def main():
    torch.manual_seed(0)  # 실행할 때마다 결과가 같도록 난수를 고정합니다.

    학습_데이터, 테스트_데이터 = 데이터_불러오기()
    학습_로더 = DataLoader(학습_데이터, batch_size=128, shuffle=True)
    테스트_로더 = DataLoader(테스트_데이터, batch_size=1000)

    모델 = DigitCNN()
    손실함수 = nn.CrossEntropyLoss()  # 정답과 얼마나 다른지 계산하는 함수
    최적화기 = torch.optim.Adam(모델.parameters(), lr=1e-3)  # 가중치를 고치는 방법

    반복_횟수 = 3  # 전체 데이터를 몇 번 반복해서 볼지 (에포크)
    for 에포크 in range(1, 반복_횟수 + 1):
        모델.train()  # 학습 모드 (드롭아웃 켜기)
        for 순번, (x, y) in enumerate(학습_로더):
            x = 이미지_흔들기(x)
            최적화기.zero_grad()          # 이전 기울기 지우기
            손실 = 손실함수(모델(x), y)   # 예측하고 틀린 정도 계산
            손실.backward()               # 각 가중치가 얼마나 틀렸는지 계산 (역전파)
            최적화기.step()               # 가중치를 조금 고치기
            if 순번 % 100 == 0:
                print(f"  에포크 {에포크} - {순번 * len(x):>5}/{len(학습_데이터)}장, 손실 {손실.item():.4f}")

        print(f"에포크 {에포크} 완료 - 테스트 정확도: {정확도_측정(모델, 테스트_로더):.2f}%")

    torch.save(모델.state_dict(), 모델_경로)
    print(f"학습된 모델을 저장했습니다: {모델_경로}")


if __name__ == "__main__":
    main()
