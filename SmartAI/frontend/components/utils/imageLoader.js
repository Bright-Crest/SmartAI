/**
 * 图片预加载工具
 * 用于提前加载图片资源，避免渲染时出现闪烁
 */

/**
 * 预加载图片
 * @param {string[]} imageUrls - 图片URL数组
 * @returns {Promise<HTMLImageElement[]>} - 加载完成的图片元素数组
 */
export const preloadImages = imageUrls => {
  if (!imageUrls || !Array.isArray(imageUrls) || imageUrls.length === 0) {
    return Promise.resolve([]);
  }

  const imagePromises = imageUrls.map(url => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
      img.src = url;
    });
  });

  return Promise.all(imagePromises);
};

/**
 * 检查图片是否已缓存
 * @param {string} src - 图片URL
 * @returns {boolean} - 是否已缓存
 */
export const isImageCached = src => {
  const img = new Image();
  img.src = src;
  return img.complete;
};

/**
 * 获取图片尺寸
 * @param {string} src - 图片URL
 * @returns {Promise<{width: number, height: number}>} - 图片尺寸
 */
export const getImageDimensions = src => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({
        width: img.width,
        height: img.height,
      });
    };
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
};
