/**
 * 格式化工具函数
 */

/**
 * 格式化数值，添加千位分隔符
 *
 * @param {number} value - 要格式化的数值
 * @param {number} [decimals=0] - 小数位数
 * @returns {string} 格式化后的数值字符串
 */
export function formatNumber(value, decimals = 0) {
  if (value === null || value === undefined) return '-';

  try {
    return Number(value).toLocaleString('zh-CN', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  } catch (error) {
    console.error('格式化数字错误:', error);
    return String(value);
  }
}

/**
 * 格式化货币值，添加货币符号
 *
 * @param {number} value - 要格式化的货币值
 * @param {string} [currency='ETH'] - 货币类型
 * @param {number} [decimals=4] - 小数位数
 * @returns {string} 格式化后的货币字符串
 */
export function formatCurrency(value, currency = 'ETH', decimals = 4) {
  if (value === null || value === undefined) return '-';

  try {
    if (value === 0) return `0 ${currency}`;

    // 处理小数位数
    const formattedValue = Number(value).toLocaleString('zh-CN', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });

    return `${formattedValue} ${currency}`;
  } catch (error) {
    console.error('格式化货币错误:', error);
    return `${value} ${currency}`;
  }
}

/**
 * 缩短地址，仅显示开头和结尾部分
 *
 * @param {string} address - 区块链地址
 * @param {number} [startChars=6] - 保留开头字符数
 * @param {number} [endChars=4] - 保留结尾字符数
 * @returns {string} 缩短后的地址
 */
export function shortenAddress(address, startChars = 6, endChars = 4) {
  if (!address) return '';
  if (address.length <= startChars + endChars) return address;

  return `${address.substring(0, startChars)}...${address.substring(address.length - endChars)}`;
}

/**
 * 格式化日期时间
 *
 * @param {string|Date} datetime - 日期时间
 * @param {boolean} [includeSeconds=false] - 是否包含秒数
 * @returns {string} 格式化后的日期时间字符串
 */
export function formatDateTime(datetime, includeSeconds = false) {
  if (!datetime) return '';

  try {
    const date = typeof datetime === 'string' ? new Date(datetime) : datetime;

    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: includeSeconds ? '2-digit' : undefined,
      hour12: false,
    });
  } catch (error) {
    console.error('格式化日期时间错误:', error);
    return String(datetime);
  }
}

/**
 * 格式化日期
 *
 * @param {string|Date} date - 日期
 * @returns {string} 格式化后的日期字符串
 */
export function formatDate(date) {
  if (!date) return '';

  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;

    return dateObj.toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    });
  } catch (error) {
    console.error('格式化日期错误:', error);
    return String(date);
  }
}

/**
 * 格式化时间段
 *
 * @param {number} seconds - 时间秒数
 * @returns {string} 格式化后的时间段
 */
export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return '-';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}小时 ${minutes}分钟`;
  } else if (minutes > 0) {
    return `${minutes}分钟 ${remainingSeconds}秒`;
  } else {
    return `${remainingSeconds}秒`;
  }
}

/**
 * 将字节大小转换为可读格式（KB, MB, GB等）
 * @param {number} bytes - 字节数
 * @param {number} decimals - 保留小数位数
 * @returns {string} 格式化后的大小字符串
 */
export function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
}

/**
 * 截断长文本并添加省略号
 * @param {string} text - 需要截断的文本
 * @param {number} maxLength - 最大长度
 * @returns {string} 截断后的文本
 */
export function truncateText(text, maxLength = 50) {
  if (!text || text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}

/**
 * 格式化区块链地址，显示开头和结尾，中间部分用省略号代替
 * @param {string} address - 区块链地址
 * @param {number} startChars - 开头显示的字符数
 * @param {number} endChars - 结尾显示的字符数
 * @returns {string} 格式化后的地址
 */
export function formatBlockchainAddress(address, startChars = 6, endChars = 4) {
  if (!address) return '';
  if (address.length <= startChars + endChars) return address;

  return address.substring(0, startChars) + '...' + address.substring(address.length - endChars);
}

/**
 * 截断地址
 * @param {string} address - 区块链地址
 * @returns {string} 截断后的地址
 */
export function truncateAddress(address) {
  return formatBlockchainAddress(address);
}

// 导出所有工具函数
export default {
  formatDateTime,
  formatNumber,
  formatCurrency,
  formatBytes,
  truncateText,
  formatBlockchainAddress,
  formatDate,
  truncateAddress,
  formatDuration,
};
