@echo off
rem 손글씨 숫자 인식 프로그램의 웹 버전을 켭니다. 브라우저가 자동으로 열립니다.
rem 이 창을 닫거나 Ctrl+C를 누르면 웹 버전이 꺼집니다.
cd /d "%~dp0"
".venv\Scripts\python.exe" web.py
pause
