#!/bin/bash

# 设置彩色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未安装。请安装后重试。"
        return 1
    else
        print_success "$1 已安装: $(command -v $1)"
        return 0
    fi
}

# 检查必要的依赖
check_dependencies() {
    print_info "正在检查必要的依赖..."
    
    local missing_deps=0
    
    # 检查Node.js (>=16)
    if check_command node; then
        node_version=$(node -v | cut -d 'v' -f 2)
        node_major_version=$(echo $node_version | cut -d '.' -f 1)
        if [ "$node_major_version" -lt 16 ]; then
            print_warning "Node.js版本过低: $node_version, 建议更新到v16或更高版本"
        else
            print_success "Node.js 版本: $node_version"
        fi
    else
        missing_deps=$((missing_deps+1))
    fi
    
    # 检查Yarn
    if check_command yarn; then
        yarn_version=$(yarn -v)
        print_success "Yarn 版本: $yarn_version"
    else
        print_warning "Yarn未安装。将尝试使用npm安装"
        if check_command npm; then
            print_info "尝试安装Yarn..."
            npm install -g yarn
            if [ $? -eq 0 ]; then
                print_success "Yarn安装成功"
            else
                print_error "Yarn安装失败"
                missing_deps=$((missing_deps+1))
            fi
        else
            missing_deps=$((missing_deps+1))
        fi
    fi
    
    # 检查Git
    if check_command git; then
        git_version=$(git --version | awk '{print $3}')
        print_success "Git 版本: $git_version"
    else
        missing_deps=$((missing_deps+1))
    fi
    
    # 检查Docker (可选)
    if check_command docker; then
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
        print_success "Docker 版本: $docker_version"
        
        # 检查Docker Compose
        if check_command docker-compose; then
            docker_compose_version=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
            print_success "Docker Compose 版本: $docker_compose_version"
        else
            print_warning "Docker Compose未安装。如果需要使用Docker启动，建议安装"
        fi
    else
        print_warning "Docker未安装。如果需要使用Docker启动，建议安装"
    fi
    
    if [ $missing_deps -gt 0 ]; then
        print_error "存在 $missing_deps 个必需依赖未安装。请安装后重试。"
        exit 1
    fi
    
    print_success "所有必需依赖已安装!"
}

# 设置环境变量
setup_env() {
    print_info "正在设置环境变量..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "已从.env.example创建.env文件"
        else
            cat > .env << EOF
# SmartAI环境配置
NODE_ENV=development

# 前端配置
NEXT_PUBLIC_API_URL=http://localhost:3000

# 后端配置
PORT=3000
DATABASE_URL=mongodb://localhost:27017/smartai
REDIS_HOST=localhost
REDIS_PORT=6379

# 区块链配置
ETHERSCAN_API_KEY=
WEB3_PROVIDER_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY
EOF
            print_success "已创建默认.env文件"
        fi
        
        print_warning "请编辑.env文件，填入必要的API密钥和配置"
    else
        print_info ".env文件已存在，跳过创建"
    fi
}

# 安装依赖
install_dependencies() {
    print_info "正在安装项目依赖..."
    
    # 安装根目录依赖
    print_info "安装根目录依赖..."
    yarn install
    if [ $? -ne 0 ]; then
        print_error "根目录依赖安装失败"
        exit 1
    fi
    print_success "根目录依赖安装成功"
    
    # 安装前端依赖
    if [ -d "frontend" ]; then
        print_info "安装前端依赖..."
        cd frontend
        yarn install
        if [ $? -ne 0 ]; then
            print_error "前端依赖安装失败"
            cd ..
            exit 1
        fi
        cd ..
        print_success "前端依赖安装成功"
    else
        print_warning "未找到frontend目录，跳过前端依赖安装"
    fi
    
    # 安装后端依赖
    if [ -d "backend" ]; then
        print_info "安装后端依赖..."
        cd backend
        yarn install
        if [ $? -ne 0 ]; then
            print_error "后端依赖安装失败"
            cd ..
            exit 1
        fi
        cd ..
        print_success "后端依赖安装成功"
    else
        print_warning "未找到backend目录，跳过后端依赖安装"
    fi
}

# 启动开发服务器
start_dev() {
    print_info "启动开发服务器的方法:"
    echo ""
    echo "  1. 使用yarn命令启动:"
    echo "     # 启动前端服务"
    echo "     cd frontend && yarn dev"
    echo ""
    echo "     # 启动后端服务 (新终端)"
    echo "     cd backend && yarn dev"
    echo ""
    echo "  2. 使用Docker启动 (如果已安装Docker):"
    echo "     docker-compose up -d"
    echo ""
    
    # 询问是否要自动启动
    read -p "是否要现在启动服务? (y/n): " autostart
    
    if [[ $autostart =~ ^[Yy]$ ]]; then
        if command -v docker &> /dev/null && [ -f "docker-compose.yml" ]; then
            print_info "使用Docker启动服务..."
            docker-compose up -d
            if [ $? -eq 0 ]; then
                print_success "服务启动成功!"
                print_info "前端访问: http://localhost:3000"
                print_info "后端访问: http://localhost:3000/api"
            else
                print_error "Docker服务启动失败，请检查错误信息"
                
                print_info "尝试使用yarn启动..."
                # 启动后端
                if [ -d "backend" ]; then
                    print_info "启动后端服务..."
                    cd backend
                    yarn dev &
                    backend_pid=$!
                    cd ..
                    print_success "后端服务启动成功，PID: $backend_pid"
                fi
                
                # 启动前端
                if [ -d "frontend" ]; then
                    print_info "启动前端服务..."
                    cd frontend
                    yarn dev &
                    frontend_pid=$!
                    cd ..
                    print_success "前端服务启动成功，PID: $frontend_pid"
                fi
                
                print_info "服务已启动!"
                print_info "前端访问: http://localhost:3000"
                print_info "后端访问: http://localhost:3000/api"
                print_info "使用 'kill $backend_pid $frontend_pid' 停止服务"
            fi
        else
            # 启动后端
            if [ -d "backend" ]; then
                print_info "启动后端服务..."
                cd backend
                yarn dev &
                backend_pid=$!
                cd ..
                print_success "后端服务启动成功，PID: $backend_pid"
            fi
            
            # 启动前端
            if [ -d "frontend" ]; then
                print_info "启动前端服务..."
                cd frontend
                yarn dev &
                frontend_pid=$!
                cd ..
                print_success "前端服务启动成功，PID: $frontend_pid"
            fi
            
            print_info "服务已启动!"
            print_info "前端访问: http://localhost:3000"
            print_info "后端访问: http://localhost:3000/api"
            print_info "使用 'kill $backend_pid $frontend_pid' 停止服务"
        fi
    else
        print_info "请按照上述说明手动启动服务"
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "     SmartAI 项目一键安装与配置脚本     "
    echo "========================================"
    echo ""
    
    # 检查依赖
    check_dependencies
    
    # 设置环境变量
    setup_env
    
    # 安装依赖
    install_dependencies
    
    # 提供启动说明
    start_dev
    
    echo ""
    print_success "安装与配置完成!"
    echo "可以在项目文档中查找更多信息:"
    echo "- 项目架构: docs/ARCHITECTURE.md"
    echo "- API文档: docs/api.md"
    echo "- 配置指南: docs/configuration.md"
    echo "- CI/CD文档: docs/CI_CD.md"
    echo ""
    echo "如有问题，请访问 GitHub Issues 页面寻求帮助"
    echo "========================================"
}

# 执行主函数
main 