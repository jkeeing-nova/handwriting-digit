"""
손글씨 숫자를 인식하는 CNN(합성곱 신경망) 모델 구조를 정의합니다.
train.py(학습)와 app.py(인식 프로그램)가 함께 사용합니다.
"""

from torch import nn


class DigitCNN(nn.Module):
    """28x28 흑백 이미지를 받아서 0~9 각 숫자일 점수 10개를 내놓는 모델입니다."""

    def __init__(self):
        super().__init__()
        # 특징 추출 부분: 이미지에서 선, 곡선 같은 모양을 찾아냅니다.
        self.특징추출 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 3x3 필터 32개로 단순한 모양(선, 점) 찾기
            nn.ReLU(),                                     # 음수는 0으로 (활성화 함수)
            nn.MaxPool2d(2),                               # 28x28 -> 14x14로 줄이기
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 필터 64개로 더 복잡한 모양(고리, 꺾임) 찾기
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14 -> 7x7로 줄이기
        )
        # 분류 부분: 찾아낸 특징을 보고 어떤 숫자인지 판단합니다.
        self.분류 = nn.Sequential(
            nn.Flatten(),                 # 64x7x7 덩어리를 한 줄(3136개)로 펴기
            nn.Linear(64 * 7 * 7, 128),   # 3136개 -> 128개
            nn.ReLU(),
            nn.Dropout(0.3),              # 학습 중 일부 뉴런을 꺼서 외워버리는 것(과적합) 막기
            nn.Linear(128, 10),           # 128개 -> 숫자 0~9 점수 10개
        )

    def forward(self, x):
        """이미지를 넣으면 숫자별 점수를 계산합니다."""
        return self.분류(self.특징추출(x))
