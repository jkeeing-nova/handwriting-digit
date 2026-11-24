r"""
마우스로 숫자를 쓰면 학습된 CNN 모델이 어떤 숫자인지 알아맞히는 프로그램입니다.

실행 방법:
    .venv\Scripts\python.exe app.py
(먼저 train.py를 실행해서 mnist_cnn.pt 파일을 만들어 두어야 합니다.)
웹 브라우저에서 쓰려면 web.py를 실행합니다 (README.md 참고).

사용 방법:
    - 왼쪽 검은 칸에 마우스 왼쪽 버튼을 누른 채로 숫자를 씁니다.
    - 손을 떼면 자동으로 인식합니다.
    - [지우기] 버튼이나 마우스 오른쪽 버튼으로 칸을 비웁니다.
"""

import tkinter as tk

import numpy as np
from PIL import ImageDraw

# 인식 기능은 웹 버전(web.py)과 함께 쓰는 recognizer.py에 있습니다.
from recognizer import 캔버스_크기, 펜_굵기, 모델_불러오기, 빈_그림_만들기, 숫자_인식하기


class 손글씨_인식_프로그램:
    """그림판, 결과 표시, 숫자별 확률 막대를 가진 화면입니다."""

    def __init__(self, 창):
        self.모델 = 모델_불러오기()
        self.창 = 창
        창.title("손글씨 숫자 인식 (MNIST CNN)")
        창.resizable(False, False)

        # 실제 인식에 쓰는 그림 (화면 그림판과 똑같이 따라 그립니다)
        self.그림 = 빈_그림_만들기()
        self.붓 = ImageDraw.Draw(self.그림)
        self.이전_점 = None

        # ----- 왼쪽: 그림판 -----
        왼쪽_영역 = tk.Frame(창, padx=12, pady=12)
        왼쪽_영역.grid(row=0, column=0, sticky="n")
        tk.Label(왼쪽_영역, text="여기에 숫자 하나를 쓰세요", font=("맑은 고딕", 11)).pack(anchor="w")
        self.캔버스 = tk.Canvas(왼쪽_영역, width=캔버스_크기, height=캔버스_크기,
                              bg="black", cursor="pencil", highlightthickness=0)
        self.캔버스.pack(pady=(6, 8))
        tk.Button(왼쪽_영역, text="지우기", font=("맑은 고딕", 11), width=10,
                  command=self.지우기).pack()

        # 마우스 동작 연결
        self.캔버스.bind("<ButtonPress-1>", self.그리기_시작)
        self.캔버스.bind("<B1-Motion>", self.그리기)
        self.캔버스.bind("<ButtonRelease-1>", self.그리기_끝)
        self.캔버스.bind("<Button-3>", lambda 이벤트: self.지우기())

        # ----- 오른쪽: 결과 -----
        오른쪽_영역 = tk.Frame(창, padx=12, pady=12)
        오른쪽_영역.grid(row=0, column=1, sticky="n")
        tk.Label(오른쪽_영역, text="인식 결과", font=("맑은 고딕", 11)).pack(anchor="w")
        self.결과_글자 = tk.Label(오른쪽_영역, text="?", font=("맑은 고딕", 64, "bold"), width=2)
        self.결과_글자.pack()
        self.확신도_글자 = tk.Label(오른쪽_영역, text="", font=("맑은 고딕", 10), fg="gray")
        self.확신도_글자.pack(pady=(0, 8))

        # 숫자 0~9 각각의 확률을 막대로 보여 줍니다.
        self.막대_캔버스 = tk.Canvas(오른쪽_영역, width=220, height=200, highlightthickness=0)
        self.막대_캔버스.pack()

        # 모델이 실제로 보는 28x28 이미지를 확대해서 보여 줍니다.
        tk.Label(오른쪽_영역, text="모델이 보는 28x28 이미지", font=("맑은 고딕", 9),
                 fg="gray").pack(anchor="w", pady=(8, 2))
        self.미리보기 = tk.Canvas(오른쪽_영역, width=84, height=84, bg="black", highlightthickness=0)
        self.미리보기.pack(anchor="w")

        self.막대_그리기(np.zeros(10))

    # ----- 그리기 -----
    def 그리기_시작(self, 이벤트):
        self.이전_점 = (이벤트.x, 이벤트.y)
        self.점_찍기(이벤트.x, 이벤트.y)

    def 그리기(self, 이벤트):
        x, y = 이벤트.x, 이벤트.y
        if self.이전_점:
            # 화면과 인식용 그림 양쪽에 같은 선을 그립니다.
            self.캔버스.create_line(*self.이전_점, x, y, fill="white", width=펜_굵기,
                                  capstyle=tk.ROUND, smooth=True)
            self.붓.line([self.이전_점, (x, y)], fill=255, width=펜_굵기)
        self.점_찍기(x, y)
        self.이전_점 = (x, y)

    def 점_찍기(self, x, y):
        """선 끝을 둥글게 만들기 위해 동그라미를 찍습니다."""
        r = 펜_굵기 / 2
        self.캔버스.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.붓.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def 그리기_끝(self, 이벤트):
        self.이전_점 = None
        self.인식하기()  # 손을 떼면 바로 인식합니다.

    def 지우기(self):
        self.캔버스.delete("all")
        self.붓.rectangle([0, 0, 캔버스_크기, 캔버스_크기], fill=0)
        self.결과_글자.config(text="?")
        self.확신도_글자.config(text="")
        self.미리보기.delete("all")
        self.막대_그리기(np.zeros(10))

    # ----- 인식 -----
    def 인식하기(self):
        작은_그림, 확률 = 숫자_인식하기(self.모델, self.그림)
        if 작은_그림 is None:
            return

        숫자 = int(확률.argmax())
        self.결과_글자.config(text=str(숫자))
        self.확신도_글자.config(text=f"확신도 {확률[숫자] * 100:.1f}%")
        self.막대_그리기(확률)
        self.미리보기_그리기(작은_그림)

    def 막대_그리기(self, 확률):
        """숫자 0~9의 확률을 가로 막대그래프로 그립니다."""
        c = self.막대_캔버스
        c.delete("all")
        최고 = int(확률.argmax()) if 확률.sum() > 0 else -1
        for 숫자 in range(10):
            y = 숫자 * 20 + 2
            길이 = 확률[숫자] * 160
            색 = "#2b6cb0" if 숫자 == 최고 else "#a0aec0"
            c.create_text(10, y + 8, text=str(숫자), font=("맑은 고딕", 10))
            c.create_rectangle(24, y + 2, 24 + 160, y + 14, fill="#edf2f7", outline="")
            c.create_rectangle(24, y + 2, 24 + 길이, y + 14, fill=색, outline="")
            c.create_text(190, y + 8, text=f"{확률[숫자] * 100:.0f}%", anchor="w",
                          font=("맑은 고딕", 8), fill="gray")

    def 미리보기_그리기(self, 작은_그림):
        """28x28 이미지를 3배로 확대해서 보여 줍니다."""
        self.미리보기.delete("all")
        배열 = np.array(작은_그림)
        for 세로 in range(28):
            for 가로 in range(28):
                밝기 = int(배열[세로, 가로])
                if 밝기 > 10:
                    색 = f"#{밝기:02x}{밝기:02x}{밝기:02x}"
                    self.미리보기.create_rectangle(가로 * 3, 세로 * 3, 가로 * 3 + 3, 세로 * 3 + 3,
                                                fill=색, outline="")


if __name__ == "__main__":
    창 = tk.Tk()
    손글씨_인식_프로그램(창)
    창.mainloop()
