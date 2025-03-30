/**
 * 区块链地址验证工具
 * 提供各种区块链地址格式的验证函数
 */

/**
 * 验证以太坊地址格式
 * @param address 待验证的地址
 * @returns 是否为有效地址
 */
export function isValidEthAddress(address: string): boolean {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}

/**
 * 验证比特币地址格式
 * @param address 待验证的地址
 * @returns 是否为有效地址
 */
export function isValidBtcAddress(address: string): boolean {
  // 比特币地址验证：以1、3或bc1开头
  return /^(1|3)[a-zA-Z0-9]{25,34}$/.test(address) || 
         /^(bc1)[a-zA-Z0-9]{25,90}$/.test(address);
}

/**
 * 验证区块链地址格式
 * 支持多链地址验证
 * @param address 待验证的地址
 * @returns 是否为有效地址
 */
export function isValidAddress(address: string): boolean {
  if (!address || typeof address !== 'string') {
    return false;
  }
  
  // 以太坊、智能链等以太坊兼容链地址
  if (isValidEthAddress(address)) {
    return true;
  }
  
  // 比特币地址
  if (isValidBtcAddress(address)) {
    return true;
  }
  
  // TODO: 可扩展其他链的地址验证
  
  return false;
}

/**
 * 标准化以太坊地址
 * @param address 输入地址
 * @returns 标准化后的地址
 */
export function normalizeEthAddress(address: string): string {
  // 转换为小写
  let normalized = address.toLowerCase();
  
  // 确保有0x前缀
  if (!normalized.startsWith('0x')) {
    normalized = '0x' + normalized;
  }
  
  return normalized;
}

/**
 * 标准化区块链地址
 * @param address 输入地址
 * @returns 标准化后的地址
 */
export function normalizeAddress(address: string): string {
  if (!address) {
    return '';
  }
  
  // 以太坊地址标准化
  if (isValidEthAddress(address) || address.startsWith('0x')) {
    return normalizeEthAddress(address);
  }
  
  // 比特币地址不做特殊处理
  
  return address;
}

/**
 * 获取地址的链类型
 * @param address 区块链地址
 * @returns 链类型
 */
export function getAddressChainType(address: string): 'ethereum' | 'bitcoin' | 'unknown' {
  if (isValidEthAddress(address)) {
    return 'ethereum';
  }
  
  if (isValidBtcAddress(address)) {
    return 'bitcoin';
  }
  
  return 'unknown';
}

/**
 * 打码处理区块链地址，保留头尾字符
 * @param address 区块链地址
 * @param visibleChars 头尾保留的字符数
 * @returns 打码后的地址
 */
export function maskAddress(address: string, visibleChars: number = 6): string {
  if (!address || address.length <= visibleChars * 2) {
    return address;
  }
  
  const prefix = address.substring(0, visibleChars);
  const suffix = address.substring(address.length - visibleChars);
  
  return `${prefix}...${suffix}`;
} 