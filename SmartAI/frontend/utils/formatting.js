/**
 * 格式化数字，增加千位分隔符及保留小数位
 * @param {number} value - 要格式化的数字
 * @param {number} decimals - 保留的小数位数
 * @returns {string} - 格式化后的字符串
 */
export const formatNumber = (value, decimals = 2) => {
  // 处理非数字情况
  if (value === null || value === undefined || isNaN(value)) {
    return '0';
  }

  // 转换为数字类型
  const num = parseFloat(value);

  // 不同数量级使用不同格式
  if (Math.abs(num) >= 1e9) {
    // 十亿及以上使用B表示
    return (num / 1e9).toFixed(decimals) + 'B';
  } else if (Math.abs(num) >= 1e6) {
    // 百万使用M表示
    return (num / 1e6).toFixed(decimals) + 'M';
  } else if (Math.abs(num) >= 1e3) {
    // 千位使用千位分隔符
    return num.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  } else if (Math.abs(num) < 0.01 && num !== 0) {
    // 极小数使用科学计数法
    return num.toExponential(decimals);
  } else {
    // 其他情况正常显示
    return num.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }
};

/**
 * 格式化金额，增加币种符号
 * @param {number} value - 要格式化的金额
 * @param {string} currency - 币种
 * @returns {string} - 格式化后的金额字符串
 */
export const formatCurrency = (value, currency = 'USD') => {
  if (value === null || value === undefined || isNaN(value)) {
    return '$0.00';
  }

  const num = parseFloat(value);

  const symbols = {
    USD: '$',
    EUR: '€',
    GBP: '£',
    CNY: '¥',
    JPY: '¥',
    ETH: 'Ξ',
    BTC: '₿',
  };

  const symbol = symbols[currency] || '';

  // 对于ETH和BTC使用更多小数位
  const decimals = ['ETH', 'BTC'].includes(currency) ? 6 : 2;

  return `${symbol}${formatNumber(num, decimals)}`;
};

/**
 * 格式化百分比
 * @param {number} value - 百分比值（如0.15表示15%）
 * @param {number} decimals - 小数位数
 * @returns {string} - 格式化的百分比
 */
export const formatPercent = (value, decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '0%';
  }

  // 数据是小数形式的百分比（如0.15表示15%）
  const num = parseFloat(value) * 100;

  return `${num.toFixed(decimals)}%`;
};

/**
 * 格式化地址，显示前几位和后几位
 * @param {string} address - 完整地址
 * @param {number} prefix - 保留的前缀长度
 * @param {number} suffix - 保留的后缀长度
 * @returns {string} - 格式化后的地址
 */
export const formatAddress = (address, prefix = 6, suffix = 4) => {
  if (!address || typeof address !== 'string') {
    return '';
  }

  if (address.length <= prefix + suffix) {
    return address;
  }

  return `${address.slice(0, prefix)}...${address.slice(-suffix)}`;
};

/**
 * 格式化时间戳为本地时间
 * @param {number|string} timestamp - UNIX时间戳(秒)或ISO时间字符串
 * @returns {string} - 格式化的日期时间
 */
export const formatTimestamp = timestamp => {
  if (!timestamp) return '';

  let date;

  // 处理UNIX时间戳(秒)
  if (typeof timestamp === 'number' || !isNaN(parseInt(timestamp))) {
    const ts = parseInt(timestamp);
    // 如果时间戳是秒为单位，转换为毫秒
    date = new Date(ts < 1e12 ? ts * 1000 : ts);
  }
  // 处理ISO格式字符串
  else if (typeof timestamp === 'string') {
    date = new Date(timestamp);
  }

  if (isNaN(date.getTime())) {
    return '';
  }

  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
};
