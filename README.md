# 손글씨 숫자 인식 (MNIST CNN)

마우스(또는 터치)로 숫자 0~9를 쓰면 PyTorch CNN 모델이 어떤 숫자인지 알아맞힙니다.
같은 학습 모델(`mnist_cnn.pt`)을 세 가지 방법으로 쓸 수 있습니다.

| 방법 | 필요한 것 | 인식은 어디서? |
|---|---|---|
| 1. 데스크탑 프로그램 (`app.py`) | 이 컴퓨터의 Python 가상환경 | 이 컴퓨터 (Python) |
| 2. 로컬 웹 버전 (`web.py`) | 이 컴퓨터의 Python 가상환경 + 브라우저 | 이 컴퓨터 (Python 서버) |
| 3. GitHub Pages 버전 (`docs/`) | 브라우저만 (설치 없음, 휴대폰 가능) | 브라우저 안 (JavaScript) |

1번과 2번은 같은 인식 코드(`recognizer.py`)를 씁니다. 3번은 같은 처리 순서를 JavaScript(`docs/recognizer.js`)로 옮긴 것입니다.

> 📝 이 프로젝트는 Claude Code에 프롬프트를 입력해서 만들었습니다. 사용한 프롬프트는 **[만든 과정 (PROMPTS.md)](PROMPTS.md)**에 있습니다.

## 준비

- 프로젝트 가상환경 `.venv`(Python 3.12, torch CPU, numpy, pillow)가 있어야 합니다.
- 학습된 모델 파일 `mnist_cnn.pt`가 있어야 합니다. 없으면 먼저 학습합니다.

```powershell
.\.venv\Scripts\python.exe train.py
```

> 시스템의 `python` 명령은 Microsoft Store 바로가기라 동작하지 않습니다. 항상 `.venv\Scripts\python.exe`를 씁니다.
> 한글이 깨지면 먼저 `$env:PYTHONIOENCODING="utf-8"`를 실행합니다.

## 1. 데스크탑 버전 (예전과 같음)

- **`숫자인식 실행.bat`**을 더블클릭합니다 (콘솔 창 없이 열림).
- 또는 PowerShell에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe app.py
```

## 2. 웹 버전 (브라우저)

- **`웹버전 실행.bat`**을 더블클릭합니다. 검은 창(서버)이 뜨고 브라우저가 자동으로 열립니다.
- 또는 PowerShell에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe web.py
```

- 브라우저가 안 열리면 주소창에 `http://127.0.0.1:8000` 을 입력합니다.
- 끝낼 때는 검은 창에서 `Ctrl+C`를 누르거나 창을 닫습니다. **창을 닫으면 웹 화면도 더 이상 인식하지 못합니다.**
- 추가로 설치할 것은 없습니다 (파이썬 기본 웹 서버 기능만 씁니다).

옵션:

| 옵션 | 뜻 |
|---|---|
| `--port 8080` | 다른 포트 번호를 씁니다 (기본 8000). "이미 사용 중" 오류가 나면 바꿔 보세요. |
| `--host 0.0.0.0` | 같은 와이파이의 휴대폰·태블릿에서도 `http://<이 컴퓨터 IP>:8000` 으로 접속할 수 있게 합니다. Windows 방화벽 허용 창이 뜰 수 있습니다. 믿을 수 있는 네트워크에서만 쓰세요. |
| `--no-browser` | 브라우저를 자동으로 열지 않습니다. |

## 3. GitHub Pages 버전 (설치 없이 브라우저만)

- **https://jkeeing-nova.github.io/handwriting-digit/** 에 접속하면 바로 쓸 수 있습니다.
- GitHub 저장소 설정에서 Pages는 `main` 브랜치의 `/docs` 폴더로 켜져 있습니다. `docs/`를 바꿔서 올리면(push) 1~2분 뒤 반영됩니다.
- Python 서버 없이, 학습된 모델 값을 내려받아 **브라우저 안에서** 계산합니다. 그린 글씨는 어디에도 보내지 않습니다.
- 파일: `docs/index.html`(화면), `docs/recognizer.js`(인식), `docs/model.json`·`docs/model.bin`(학습된 모델 값, 약 1.6MB)
- 내 컴퓨터에서 미리 보려면 (`file://`로 직접 열면 모델 파일을 못 읽으므로 간단한 서버가 필요):

```powershell
.\.venv\Scripts\python.exe -m http.server 8001 --bind 127.0.0.1 --directory docs
```

  → 브라우저에서 `http://127.0.0.1:8001` 접속, 끝낼 때는 `Ctrl+C`

- **`train.py`로 다시 학습했다면** 모델 값을 다시 내보내고 `docs/` 변경 내용을 GitHub에 올려야 반영됩니다.

```powershell
.\.venv\Scripts\python.exe export_web_model.py
```

- 결과는 Python 버전과 같은 숫자로 나오도록 맞췄습니다. 시험용 글씨 30개에서 모두 같은 답이 나왔고, 확률 차이는 최대 0.35%였습니다. 차이는 그림을 그리고 줄이는 계산이 아주 조금 달라서 생깁니다.

## 사용 방법 (모든 버전 공통)

1. 검은 칸에 마우스 왼쪽 버튼(휴대폰은 손가락)을 누른 채로 숫자 하나를 씁니다.
2. 손을 떼면 바로 인식합니다. 오른쪽에 결과 숫자, 확신도, 0~9별 확률 막대, 모델이 실제로 보는 28×28 이미지가 나옵니다.
3. **[지우기]** 버튼이나 마우스 오른쪽 버튼으로 칸을 비웁니다.

## 파일 구성

| 파일 | 하는 일 |
|---|---|
| `model.py` | CNN 모델 구조 (`DigitCNN`) |
| `train.py` | MNIST로 학습해서 `mnist_cnn.pt`를 만듦 |
| `recognizer.py` | **공통 인식 기능**: 모델 불러오기, 그림을 MNIST 형식(28×28)으로 바꾸기, 숫자 인식, 획 좌표로 그림 다시 그리기 |
| `app.py` | 데스크탑 화면 (tkinter) |
| `web.py` | 웹 서버. 화면(`web/index.html`)을 보내 주고, `/predict`로 받은 획 좌표를 인식해서 돌려줌 |
| `web/index.html` | 웹 화면 (그림판, 결과, 확률 막대) |
| `숫자인식 실행.bat` / `웹버전 실행.bat` | 더블클릭용 실행 파일 |
| `export_web_model.py` | `mnist_cnn.pt`를 브라우저용 `docs/model.json`·`docs/model.bin`으로 내보냄 |
| `docs/` | GitHub Pages 버전 (`index.html`, `recognizer.js`, 모델 파일) |

웹 버전은 브라우저가 **그림이 아니라 획 좌표만** 서버로 보내고, 서버가 데스크탑 버전과 똑같은 방법으로 그림을 다시 그려서 인식합니다. 그래서 같은 글씨를 쓰면 두 버전의 결과가 같습니다.
