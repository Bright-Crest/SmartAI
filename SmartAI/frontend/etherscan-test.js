#!/usr/bin/env node

const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

// API 参数
const apiKey = 'W6DDB16I381Z1UVDTDAP4AV21XTRPW7AVS';
const address = '0x00000000219ab540356cBB839Cbe05303d7705Fa';

// 构建 curl 命令
const buildCurlCommand = params => {
  const queryParams = Object.entries(params)
    .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
    .join('&');

  return `curl -s "https://api.etherscan.io/api?${queryParams}"`;
};

async function testEtherscanAPI() {
  try {
    console.log('开始测试 Etherscan API 连接...');

    // 测试获取余额
    const balanceParams = {
      module: 'account',
      action: 'balance',
      address,
      tag: 'latest',
      apikey: apiKey,
    };

    const curlCmd = buildCurlCommand(balanceParams);
    console.log(`执行命令: ${curlCmd.replace(apiKey, '***HIDDEN***')}`);

    const { stdout, stderr } = await execPromise(curlCmd);

    if (stderr) {
      console.error('curl 错误:', stderr);
      return;
    }

    // 解析结果
    try {
      const result = JSON.parse(stdout);
      console.log('API 响应成功:', JSON.stringify(result, null, 2));

      if (result.status === '1') {
        const balanceInEth = parseInt(result.result) / 1e18;
        console.log(`地址余额: ${balanceInEth} ETH`);
      } else {
        console.error('API 返回错误:', result.message);
      }
    } catch (e) {
      console.error('解析响应失败:', e.message);
      console.error('原始响应:', stdout.substring(0, 200));
    }
  } catch (error) {
    console.error('测试失败:', error.message);
  }
}

// 执行测试
testEtherscanAPI();
