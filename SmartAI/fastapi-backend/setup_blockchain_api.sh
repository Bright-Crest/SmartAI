#!/bin/bash
# 区块链API安装脚本
# 用于安装和设置区块链API服务

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_green() {
    echo -e "${GREEN}$1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}$1${NC}"
}

print_red() {
    echo -e "${RED}$1${NC}"
}

print_blue() {
    echo -e "${BLUE}$1${NC}"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_red "错误: $1 命令未找到。请先安装 $1。"
        exit 1
    fi
}

# 获取当前目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 欢迎信息
print_blue "====================================================="
print_blue "       区块链情报分析平台API 安装脚本       "
print_blue "====================================================="
echo ""

# 检查Python版本
print_yellow "检查Python版本..."
check_command python3

PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $PYTHON_VERSION != *"Python 3.11"* ]]; then
    print_yellow "警告: 未检测到Python 3.11。尝试查找特定版本..."
    
    if command -v python3.11 &> /dev/null; then
        print_green "找到Python 3.11!"
        PYTHON="python3.11"
    else
        print_red "错误: 未找到Python 3.11。请安装Python 3.11后再运行此脚本。"
        print_yellow "提示: 在macOS上，可以使用 'brew install python@3.11' 安装Python 3.11。"
        print_yellow "      在Ubuntu上，可以使用 'sudo apt install python3.11' 安装Python 3.11。"
        exit 1
    fi
else
    PYTHON="python3"
fi

# 检查当前Python版本
print_green "使用Python版本: $($PYTHON --version)"

# 创建虚拟环境
print_yellow "创建虚拟环境..."
if [ -d "venv311" ]; then
    print_yellow "虚拟环境已存在，跳过创建步骤。"
else
    $PYTHON -m venv venv311
    if [ $? -ne 0 ]; then
        print_red "创建虚拟环境失败。"
        exit 1
    fi
    print_green "虚拟环境创建成功。"
fi

# 激活虚拟环境
print_yellow "激活虚拟环境..."
source venv311/bin/activate
if [ $? -ne 0 ]; then
    print_red "激活虚拟环境失败。"
    exit 1
fi
print_green "虚拟环境激活成功。"

# 安装依赖
print_yellow "安装依赖..."
pip install -r requirements-py311.txt
if [ $? -ne 0 ]; then
    print_red "安装依赖失败。"
    exit 1
fi
print_green "依赖安装成功。"

# 创建配置文件
print_yellow "创建配置文件..."
if [ ! -f ".env" ]; then
    print_yellow "未找到.env文件，创建新文件..."
    
    # 询问用户是否配置API密钥
    read -p "是否要配置以太坊节点和API密钥？(y/n): " configure_keys
    
    # 创建.env文件
    cat > .env << EOL
# 区块链情报分析平台API环境配置
# 这是一个示例配置文件，请根据实际情况修改

# 以太坊节点RPC URL
ETHEREUM_RPC_URL=

# API密钥
INFURA_API_KEY=
ETHERSCAN_API_KEY=
ALCHEMY_API_KEY=
EOL
    
    if [[ $configure_keys == "y" || $configure_keys == "Y" ]]; then
        # 询问用户输入API密钥
        read -p "请输入以太坊节点RPC URL（可选）: " ethereum_rpc_url
        read -p "请输入Infura API密钥（可选）: " infura_api_key
        read -p "请输入Etherscan API密钥（可选）: " etherscan_api_key
        read -p "请输入Alchemy API密钥（可选）: " alchemy_api_key
        
        # 更新.env文件
        sed -i.bak "s|ETHEREUM_RPC_URL=|ETHEREUM_RPC_URL=$ethereum_rpc_url|g" .env
        sed -i.bak "s|INFURA_API_KEY=|INFURA_API_KEY=$infura_api_key|g" .env
        sed -i.bak "s|ETHERSCAN_API_KEY=|ETHERSCAN_API_KEY=$etherscan_api_key|g" .env
        sed -i.bak "s|ALCHEMY_API_KEY=|ALCHEMY_API_KEY=$alchemy_api_key|g" .env
        
        # 删除备份文件
        rm -f .env.bak
        
        print_green "API密钥配置成功。"
    else
        print_yellow "跳过API密钥配置。请稍后手动编辑.env文件。"
    fi
    
    print_green ".env文件创建成功。"
else
    print_yellow ".env文件已存在，跳过创建步骤。"
    print_yellow "如需配置API密钥，请手动编辑.env文件。"
fi

# 补充说明Etherscan和Alchemy API的获取方法
print_blue "====================================================="
print_blue "       API密钥获取指南       "
print_blue "====================================================="
print_yellow "Etherscan API密钥:"
print_yellow "1. 访问 https://etherscan.io/register 注册账号"
print_yellow "2. 登录后访问 https://etherscan.io/myapikey 创建API密钥"
print_yellow ""
print_yellow "Alchemy API密钥:"
print_yellow "1. 访问 https://www.alchemy.com/ 注册账号"
print_yellow "2. 创建一个以太坊应用获取API密钥"
print_yellow ""
print_yellow "Infura API密钥:"
print_yellow "1. 访问 https://infura.io/register 注册账号"
print_yellow "2. 创建一个以太坊项目获取API密钥"
print_blue "====================================================="

# 测试API
print_yellow "是否要运行API测试？(y/n): "
read run_test
if [[ $run_test == "y" || $run_test == "Y" ]]; then
    print_yellow "启动API服务进行测试..."
    # 在后台启动API服务
    python blockchain_api.py &
    API_PID=$!
    
    # 等待API启动
    print_yellow "等待API启动..."
    sleep 3
    
    # 运行测试
    print_yellow "运行API测试..."
    python test_blockchain_api.py
    TEST_RESULT=$?
    
    # 终止API服务
    kill $API_PID
    
    if [ $TEST_RESULT -eq 0 ]; then
        print_green "API测试成功。"
    else
        print_red "API测试失败。请检查配置和依赖。"
    fi
else
    print_yellow "跳过API测试。"
fi

# 完成
print_green "====================================================="
print_green "       区块链情报分析平台API 安装完成       "
print_green "====================================================="
print_yellow "使用以下命令启动API服务:"
print_yellow "cd $(pwd)"
print_yellow "source venv311/bin/activate"
print_yellow "python blockchain_api.py"
print_yellow ""
print_yellow "API文档可在以下URL访问:"
print_yellow "Swagger UI: http://localhost:8001/docs"
print_yellow "ReDoc: http://localhost:8001/redoc"

print_blue "====================================================="
print_blue "       Etherscan/Alchemy API集成说明       "
print_blue "====================================================="
print_yellow "本API支持与Etherscan和Alchemy API集成，提供以下功能:"
print_yellow "1. Etherscan API - 查询地址余额、交易历史、合约ABI"
print_yellow "2. Alchemy API - 查询地址余额、代币余额、交易收据"
print_yellow ""
print_yellow "如果未配置相应的API密钥，API将使用模拟数据。"
print_yellow "这对开发和测试很有用，但在生产环境中建议配置正确的API密钥。"
print_blue "====================================================="

# 退出虚拟环境
deactivate

exit 0 