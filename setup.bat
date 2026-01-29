@echo off
echo 正在创建BCI项目骨架...

mkdir core 2>nul
mkdir modules 2>nul
mkdir utils 2>nul
mkdir config 2>nul
mkdir docs 2>nul

echo # BCI项目 > README.md
echo 多模态脑机接口平台 >> README.md

echo ? 骨架创建完成！
dir
pause
