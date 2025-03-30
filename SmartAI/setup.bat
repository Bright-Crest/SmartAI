@echo off
setlocal enabledelayedexpansion

:: 设置颜色
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "NC=[0m"

:: 打印带颜色的信息
:print_info
echo %BLUE%[信息]%NC% %~1
goto :eof

:print_success
echo %GREEN%[成功]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[警告]%NC% %~1
goto :eof

:print_error
echo %RED%[错误]%NC% %~1
goto :eof

echo ========================================
echo      SmartAI 项目一键安装与配置脚本     
echo ========================================
echo.

call :print_info "正在检查必要的依赖..."

:: 检查 Node.js
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "Node.js 未安装。请安装后重试: https://nodejs.org/"
    exit /b 1
) else (
    for /f "tokens=1,2,3 delims=." %%a in ('node -v') do (
        set node_version=%%a
    )
    set node_version=!node_version:~1!
    if !node_version! lss 16 (
        call :print_warning "Node.js 版本过低，建议更新到 v16 或更高版本"
    ) else (
        for /f "tokens=*" %%i in ('node -v') do (
            call :print_success "Node.js 已安装: %%i"
        )
    )
)

:: 检查 Yarn
where yarn >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_warning "Yarn 未安装。将尝试使用 npm 安装"
    call npm install -g yarn
    if %ERRORLEVEL% neq 0 (
        call :print_error "Yarn 安装失败"
        exit /b 1
    ) else (
        call :print_success "Yarn 安装成功"
    )
) else (
    for /f "tokens=*" %%i in ('yarn -v') do (
        call :print_success "Yarn 已安装: %%i"
    )
)

:: 检查 Git
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "Git 未安装。请安装后重试: https://git-scm.com/"
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('git --version') do (
        call :print_success "Git 已安装: %%i"
    )
)

:: 检查 Docker (可选)
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_warning "Docker 未安装。如果需要使用 Docker 启动，建议安装: https://www.docker.com/"
) else (
    for /f "tokens=*" %%i in ('docker --version') do (
        call :print_success "Docker 已安装: %%i"
    )
    
    :: 检查 Docker Compose
    where docker-compose >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        call :print_warning "Docker Compose 未安装。如果需要使用 Docker 启动，建议安装"
    ) else (
        for /f "tokens=*" %%i in ('docker-compose --version') do (
            call :print_success "Docker Compose 已安装: %%i"
        )
    )
)

call :print_success "所有必需依赖已安装!"

:: 设置环境变量
call :print_info "正在设置环境变量..."

if not exist .env (
    if exist .env.example (
        copy .env.example .env
        call :print_success "已从 .env.example 创建 .env 文件"
    ) else (
        (
            echo # SmartAI环境配置
            echo NODE_ENV=development
            echo.
            echo # 前端配置
            echo NEXT_PUBLIC_API_URL=http://localhost:3000
            echo.
            echo # 后端配置
            echo PORT=3000
            echo DATABASE_URL=mongodb://localhost:27017/smartai
            echo REDIS_HOST=localhost
            echo REDIS_PORT=6379
            echo.
            echo # 区块链配置
            echo ETHERSCAN_API_KEY=
            echo WEB3_PROVIDER_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY
        ) > .env
        call :print_success "已创建默认 .env 文件"
    )
    
    call :print_warning "请编辑 .env 文件，填入必要的 API 密钥和配置"
) else (
    call :print_info ".env 文件已存在，跳过创建"
)

:: 安装依赖
call :print_info "正在安装项目依赖..."

:: 安装根目录依赖
call :print_info "安装根目录依赖..."
call yarn install
if %ERRORLEVEL% neq 0 (
    call :print_error "根目录依赖安装失败"
    exit /b 1
)
call :print_success "根目录依赖安装成功"

:: 安装前端依赖
if exist frontend (
    call :print_info "安装前端依赖..."
    cd frontend
    call yarn install
    if %ERRORLEVEL% neq 0 (
        call :print_error "前端依赖安装失败"
        cd ..
        exit /b 1
    )
    cd ..
    call :print_success "前端依赖安装成功"
) else (
    call :print_warning "未找到 frontend 目录，跳过前端依赖安装"
)

:: 安装后端依赖
if exist backend (
    call :print_info "安装后端依赖..."
    cd backend
    call yarn install
    if %ERRORLEVEL% neq 0 (
        call :print_error "后端依赖安装失败"
        cd ..
        exit /b 1
    )
    cd ..
    call :print_success "后端依赖安装成功"
) else (
    call :print_warning "未找到 backend 目录，跳过后端依赖安装"
)

:: 启动开发服务器
call :print_info "启动开发服务器的方法:"
echo.
echo   1. 使用 yarn 命令启动:
echo      # 启动前端服务
echo      cd frontend ^&^& yarn dev
echo.
echo      # 启动后端服务 (新终端)
echo      cd backend ^&^& yarn dev
echo.
echo   2. 使用 Docker 启动 (如果已安装 Docker):
echo      docker-compose up -d
echo.

:: 询问是否要自动启动
set /p autostart="是否要现在启动服务? (y/n): "

if /i "%autostart%" == "y" (
    where docker >nul 2>&1 && if exist docker-compose.yml (
        call :print_info "使用 Docker 启动服务..."
        call docker-compose up -d
        
        if %ERRORLEVEL% equ 0 (
            call :print_success "服务启动成功!"
            call :print_info "前端访问: http://localhost:3000"
            call :print_info "后端访问: http://localhost:3000/api"
        ) else (
            call :print_error "Docker 服务启动失败，请检查错误信息"
            
            call :print_info "尝试使用 yarn 启动..."
            
            :: 创建启动脚本
            (
                echo @echo off
                echo.
                echo echo 启动 SmartAI 服务...
                echo.
                echo start "SmartAI 后端" cmd /c "cd backend ^&^& yarn dev"
                echo start "SmartAI 前端" cmd /c "cd frontend ^&^& yarn dev"
                echo.
                echo echo 服务已启动!
                echo echo 前端访问: http://localhost:3000
                echo echo 后端访问: http://localhost:3000/api
                echo.
                echo echo 按任意键关闭所有服务...
                echo pause ^> nul
                echo.
                echo echo 正在关闭服务...
                echo taskkill /f /im node.exe
                echo echo 服务已关闭!
            ) > start-services.bat
            
            call :print_success "创建服务启动脚本: start-services.bat"
            call :print_info "请运行 start-services.bat 启动服务"
        )
    ) else (
        :: 创建启动脚本
        (
            echo @echo off
            echo.
            echo echo 启动 SmartAI 服务...
            echo.
            echo start "SmartAI 后端" cmd /c "cd backend ^&^& yarn dev"
            echo start "SmartAI 前端" cmd /c "cd frontend ^&^& yarn dev"
            echo.
            echo echo 服务已启动!
            echo echo 前端访问: http://localhost:3000
            echo echo 后端访问: http://localhost:3000/api
            echo.
            echo echo 按任意键关闭所有服务...
            echo pause ^> nul
            echo.
            echo echo 正在关闭服务...
            echo taskkill /f /im node.exe
            echo echo 服务已关闭!
        ) > start-services.bat
        
        call :print_success "创建服务启动脚本: start-services.bat"
        call :print_info "请运行 start-services.bat 启动服务"
    )
) else (
    call :print_info "请按照上述说明手动启动服务"
)

echo.
call :print_success "安装与配置完成!"
echo 可以在项目文档中查找更多信息:
echo - 项目架构: docs\ARCHITECTURE.md
echo - API文档: docs\api.md
echo - 配置指南: docs\configuration.md
echo - CI/CD文档: docs\CI_CD.md
echo.
echo 如有问题，请访问 GitHub Issues 页面寻求帮助
echo ========================================

pause 