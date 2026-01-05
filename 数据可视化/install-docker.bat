@echo off
chcp 65001 >nul

echo ==========================================
echo     Docker Desktop 安装指南
echo ==========================================

echo 正在打开Docker Desktop下载页面...
start https://www.docker.com/products/docker-desktop/

echo.
echo 安装步骤:
echo 1. 点击 "Download for Windows" 下载Docker Desktop
echo 2. 运行下载的安装程序
echo 3. 按照安装向导完成安装
echo 4. 重启电脑
echo 5. 启动Docker Desktop
echo 6. 等待Docker完全启动 (系统托盘图标变为绿色)
echo 7. 运行 start.bat 启动项目
echo.
echo 系统要求:
echo - Windows 10 64位: 专业版、企业版或教育版
echo - 启用Hyper-V和容器功能
echo - 至少4GB RAM
echo.
echo 如果遇到问题:
echo 1. 确保启用了虚拟化功能 (BIOS设置)
echo 2. 确保启用了Hyper-V功能
echo 3. 以管理员身份运行安装程序
echo.

pause
