@echo off
rem 손글씨 숫자 인식 프로그램을 검은 명령창 없이 실행합니다.
rem %~dp0 는 이 파일이 있는 폴더 경로라서, 폴더를 옮겨도 그대로 동작합니다.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app.py"