import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * 合并Tailwind CSS类名，解决类名冲突
 * @param {...string} inputs - 要合并的类名
 * @returns {string} - 合并后的类名
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
