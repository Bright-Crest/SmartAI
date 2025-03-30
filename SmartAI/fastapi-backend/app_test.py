from flask import Flask, jsonify
from web3 import Web3

app = Flask(__name__)

# 测试Web3连接
try:
    # 使用Infura的URL（需要替换为自己的API密钥）
    infura_url = "https://mainnet.infura.io/v3/YOUR_INFURA_KEY"
    
    # 也可以使用无需API密钥的公共节点
    # w3 = Web3(Web3.HTTPProvider("https://ethereum.publicnode.com"))
    
    # 创建Web3实例
    w3 = Web3(Web3.HTTPProvider(infura_url))
    web3_connected = w3.is_connected()
except Exception as e:
    web3_connected = False
    web3_error = str(e)

@app.route('/')
def home():
    return jsonify({
        'message': '区块链Web3测试API',
        'status': 'running'
    })

@app.route('/web3-status')
def web3_status():
    if web3_connected:
        return jsonify({
            'status': 'connected',
            'blockNumber': w3.eth.block_number
        })
    else:
        return jsonify({
            'status': 'disconnected',
            'error': web3_error if 'web3_error' in locals() else 'Unknown error'
        })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'dependencies': {
            'web3': 'available' if 'web3_connected' in globals() else 'imported'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 