# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

MNIST로 학습한 PyTorch CNN이 마우스로 쓴 손글씨 숫자(0~9)를 인식하는 학습용 프로젝트입니다. 사용자는 머신러닝을 처음 공부하는 단계이며, 설명과 답변은 한국어로 합니다.

## 코드 작성 규칙

- **모든 주석과 docstring은 한글로** 작성합니다(사용자 요청).
- 변수·함수·클래스 이름도 기존 코드처럼 한글 식별자를 씁니다(예: `모델_불러오기`, `손글씨_인식_프로그램`).
- `.venv\Scripts\...` 같은 백슬래시 경로가 들어간 모듈 docstring은 `r"""`로 시작해야 SyntaxWarning이 나지 않습니다.
- `숫자인식 실행.bat`은 한글 주석이 cmd에서 깨지지 않도록 **CP949(EUC-KR)로 저장**되어 있습니다. UTF-8로 다시 저장하지 마세요. 수정할 때는 PowerShell에서 `[System.Text.Encoding]::GetEncoding(949)`로 써야 합니다.

## 명령어

시스템 PATH의 `python`은 Microsoft Store 바로가기일 뿐이라 실행되지 않습니다. 항상 프로젝트 가상환경을 씁니다(Python 3.12, torch CPU 버전, numpy, pillow). 폴더 이름에 공백이 있으니 경로는 따옴표로 감쌉니다.

```powershell
# 학습: data/mnist.npz가 없으면 내려받은 뒤 학습하고 mnist_cnn.pt를 덮어씁니다 (CPU로 수 분)
.\.venv\Scripts\python.exe train.py

# 인식 프로그램 실행 (콘솔 창 없이 띄우려면 pythonw.exe 또는 "숫자인식 실행.bat")
.\.venv\Scripts\python.exe app.py
```

한글 출력이 깨지면 먼저 `$env:PYTHONIOENCODING="utf-8"`를 설정합니다. 테스트, 린트, 빌드 설정은 없습니다.

## 구조

- 실행 방법은 두 가지입니다: 데스크탑 `app.py`(tkinter), 웹 `web.py`(표준 라이브러리 `http.server`, 기본 `http://127.0.0.1:8000`, 화면은 `web/index.html`). 자세한 실행법은 `README.md`.
- 인식 로직(모델 불러오기, `mnist_형식으로_변환`, `숫자_인식하기`, 펜 굵기·캔버스 크기)은 **`recognizer.py` 한 곳**에 있고 두 버전이 함께 씁니다. 웹 화면은 획 좌표만 `/predict`로 보내고 서버가 `획으로_그림_만들기`로 데스크탑과 똑같이 다시 그립니다. 펜 굵기·캔버스 크기를 바꾸면 `web/index.html`의 같은 상수도 맞춥니다.
- GitHub Pages 버전은 `docs/`(브라우저 안에서 JavaScript로 인식, 서버 없음). `docs/recognizer.js`는 `recognizer.py`의 처리(그리기 → PIL GaussianBlur(1) 방식 상자 흐림 3번 → 자르기 → PIL LANCZOS 방식 축소 → 무게중심 이동 → CNN)를 그대로 옮긴 것이므로, Python 쪽 전처리나 `model.py` 구조를 바꾸면 이 파일도 같이 고칩니다. 가중치는 `export_web_model.py`가 `docs/model.json`·`model.bin`으로 내보내며, 다시 학습하면 이것도 다시 실행합니다. Node.js(v26)가 있어 `require("./docs/recognizer.js")`로 Python 결과와 비교 시험할 수 있습니다.
- `.gitattributes`에서 `*.bat -text`로 CP949·CRLF가 바뀌지 않게 막아 두었습니다.
- `웹버전 실행.bat`도 `숫자인식 실행.bat`처럼 CP949로 저장되어 있습니다.
- `model.py`의 `DigitCNN`을 `train.py`(학습)와 `recognizer.py`(추론)가 함께 씁니다. **모델 구조를 바꾸면 `mnist_cnn.pt`와 맞지 않게 되므로** 반드시 `train.py`를 다시 실행해야 `app.py`·`web.py`가 열립니다.
- 입력 형식은 양쪽에서 같아야 합니다: `(N, 1, 28, 28)` float 텐서, 값 범위 0~1, **검은 바탕(0)에 흰 글씨(1)**.
- 인식 정확도는 `recognizer.py`의 `mnist_형식으로_변환`이 핵심입니다. 그림판(280×280, 펜 굵기 22) 그림을 흐리게 한 뒤, 글씨 부분만 잘라 긴 변을 20px로 줄이고, 28×28 가운데에 붙인 다음 **무게중심을 (14, 14)로 옮깁니다**. MNIST 원본이 이렇게 만들어졌기 때문이며, 이 단계를 바꾸면 실제 손글씨 인식률이 크게 떨어집니다.
- `train.py`의 `이미지_흔들기`는 학습 배치마다 회전(약 ±8.5°), 크기(±10%), 이동(약 ±3px)을 무작위로 적용하는 데이터 증강입니다. 마우스로 쓴 글씨에 잘 맞도록 넣은 것입니다.
- 현재 결과: 3 에포크 학습 시 MNIST 테스트 정확도 약 98.9%.
