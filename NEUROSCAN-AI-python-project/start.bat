@echo off
chcp 65001 >nul
title NeuroScan AI - Medical Workstation
echo =====================================================================
echo        🧠 NeuroScan AI - سامانه هوشمند تشخیص تومور مغزی
echo             موتور پردازش تصویر OpenCV 5.0 + یادگیری عمیق
echo =====================================================================
echo.
echo در حال راه‌اندازی سرور ایستگاه کاری رادیولوژی روی پورت 8080...
echo پس از مشاهده پیام Ready، مرورگر خود را باز کرده و به آدرس زیر بروید:
echo.
echo    👉 http://localhost:8080
echo.
python server.py 8080
pause

