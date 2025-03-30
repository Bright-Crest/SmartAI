#!/usr/bin/env node

/**
 * 颜色对比度检查工具
 *
 * 此脚本用于检查项目中的CSS文件和组件，找出潜在的颜色对比度问题。
 * 它会扫描CSS类名和内联样式，提取颜色值，然后计算对比度，
 * 如果对比度不满足WCAG AA标准，则会报告问题。
 *
 * 用法: node scripts/check-color-contrast.js
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// WCAG 2.1 AA 标准要求的最小对比度
const MIN_CONTRAST_RATIO_NORMAL_TEXT = 4.5; // 普通文本（小于18pt或14pt粗体）
const MIN_CONTRAST_RATIO_LARGE_TEXT = 3.0; // 大文本（至少18pt或14pt粗体）

// 颜色名称到十六进制的映射
const COLOR_NAMES = {
  black: '#000000',
  white: '#FFFFFF',
  red: '#FF0000',
  green: '#008000',
  blue: '#0000FF',
  yellow: '#FFFF00',
  gray: '#808080',
  grey: '#808080',
  // 添加更多常见颜色
};

// Tailwind颜色类名映射（简化版本，实际上需要更完整的映射）
const TAILWIND_COLORS = {
  'gray-100': '#f7fafc',
  'gray-200': '#edf2f7',
  'gray-300': '#e2e8f0',
  'gray-400': '#cbd5e0',
  'gray-500': '#a0aec0',
  'gray-600': '#718096',
  'gray-700': '#4a5568',
  'gray-800': '#2d3748',
  'gray-900': '#1a202c',

  'red-100': '#fff5f5',
  'red-500': '#f56565',
  'red-700': '#c53030',
  'red-800': '#9b2c2c',

  'green-100': '#f0fff4',
  'green-500': '#48bb78',
  'green-700': '#2f855a',
  'green-800': '#276749',

  'blue-100': '#ebf8ff',
  'blue-500': '#4299e1',
  'blue-600': '#3182ce',
  'blue-700': '#2b6cb0',
  'blue-800': '#2c5282',

  'yellow-100': '#fffff0',
  'yellow-500': '#ecc94b',
  'yellow-700': '#b7791f',
  'yellow-800': '#975a16',

  'orange-100': '#fffaf0',
  'orange-500': '#ed8936',
  'orange-700': '#c05621',
  'orange-800': '#9c4221',
};

// 将rgb颜色转换为十六进制格式
function rgbToHex(r, g, b) {
  return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

// 将颜色转换为RGB对象
function parseColor(color) {
  // 如果是命名颜色，转换为十六进制
  if (COLOR_NAMES[color.toLowerCase()]) {
    color = COLOR_NAMES[color.toLowerCase()];
  }

  // 解析十六进制颜色
  if (color.startsWith('#')) {
    const hex = color.substring(1);
    if (hex.length === 3) {
      return {
        r: parseInt(hex[0] + hex[0], 16),
        g: parseInt(hex[1] + hex[1], 16),
        b: parseInt(hex[2] + hex[2], 16),
      };
    } else if (hex.length === 6) {
      return {
        r: parseInt(hex.substring(0, 2), 16),
        g: parseInt(hex.substring(2, 4), 16),
        b: parseInt(hex.substring(4, 6), 16),
      };
    }
  }

  // 解析rgb格式
  const rgbMatch = color.match(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/i);
  if (rgbMatch) {
    return {
      r: parseInt(rgbMatch[1]),
      g: parseInt(rgbMatch[2]),
      b: parseInt(rgbMatch[3]),
    };
  }

  // 解析rgba格式
  const rgbaMatch = color.match(/rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)/i);
  if (rgbaMatch) {
    return {
      r: parseInt(rgbaMatch[1]),
      g: parseInt(rgbaMatch[2]),
      b: parseInt(rgbaMatch[3]),
      a: parseFloat(rgbaMatch[4]),
    };
  }

  return null;
}

// 计算相对亮度（sRGB色彩空间）
function getLuminance(color) {
  if (!color) return 0;

  // 将RGB值标准化为[0,1]范围
  let r = color.r / 255;
  let g = color.g / 255;
  let b = color.b / 255;

  // 应用sRGB转换
  r = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
  g = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
  b = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);

  // 计算相对亮度
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// 计算两个颜色之间的对比度
function getContrastRatio(color1, color2) {
  const lum1 = getLuminance(color1);
  const lum2 = getLuminance(color2);

  // 确保亮度较高的值在分子中
  const lighter = Math.max(lum1, lum2);
  const darker = Math.min(lum1, lum2);

  return (lighter + 0.05) / (darker + 0.05);
}

// 从组件文件中提取颜色信息
function extractColorsFromFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const colors = [];

  // 查找类名中的颜色信息
  const tailwindClasses = content.match(/className\s*=\s*["']([^"']+)["']/g) || [];

  tailwindClasses.forEach(classDecl => {
    const classes = classDecl
      .replace(/className\s*=\s*["']/, '')
      .replace(/["']$/, '')
      .split(/\s+/);

    classes.forEach(cls => {
      // 提取Tailwind颜色类
      if (/^(bg|text|border)-(red|green|blue|gray|grey|yellow|orange)-\d+$/.test(cls)) {
        const parts = cls.split('-');
        const type = parts[0]; // bg, text, border
        const colorName = `${parts[1]}-${parts[2]}`;

        if (TAILWIND_COLORS[colorName]) {
          colors.push({
            type,
            color: TAILWIND_COLORS[colorName],
            class: cls,
            line: content.substring(0, content.indexOf(classDecl)).split('\n').length,
          });
        }
      }
    });
  });

  // 提取内联样式中的颜色信息
  const styleDeclarations = content.match(/style\s*=\s*{[^}]+}/g) || [];

  styleDeclarations.forEach(styleDecl => {
    // 提取颜色属性
    const colorProps =
      styleDecl.match(/(color|backgroundColor|borderColor)\s*:\s*['"]?([^'"}, ]+)['"]?/g) || [];

    colorProps.forEach(prop => {
      const match = prop.match(/(color|backgroundColor|borderColor)\s*:\s*['"]?([^'"}, ]+)['"]?/);
      if (match) {
        const type =
          match[1] === 'color' ? 'text' : match[1] === 'backgroundColor' ? 'bg' : 'border';
        const colorValue = match[2];

        colors.push({
          type,
          color: colorValue,
          line: content.substring(0, content.indexOf(styleDecl)).split('\n').length,
        });
      }
    });
  });

  return colors;
}

// 检查颜色对比度
function checkColorContrast(colors, filePath) {
  const issues = [];

  // 找到背景色和文本色的组合
  const backgrounds = colors.filter(c => c.type === 'bg');
  const texts = colors.filter(c => c.type === 'text');

  // 如果没有找到背景色，假设是白色背景
  if (backgrounds.length === 0) {
    backgrounds.push({
      type: 'bg',
      color: '#ffffff',
      default: true,
    });
  }

  // 检查每个文本颜色与每个背景色的对比度
  texts.forEach(text => {
    const textColor = parseColor(text.color);
    if (!textColor) return;

    backgrounds.forEach(bg => {
      const bgColor = parseColor(bg.color);
      if (!bgColor) return;

      const contrast = getContrastRatio(textColor, bgColor);

      if (contrast < MIN_CONTRAST_RATIO_NORMAL_TEXT) {
        issues.push({
          file: filePath,
          line: text.line || 'unknown',
          textColor: text.color,
          textClass: text.class,
          bgColor: bg.color,
          bgClass: bg.class,
          contrast,
          recommendation: contrast < 3 ? '严重问题，必须修复' : '需要改进',
        });
      }
    });
  });

  return issues;
}

// 主函数
function main() {
  // 获取组件文件列表
  const componentFiles = glob.sync('components/**/*.{jsx,tsx}', {
    cwd: path.resolve(__dirname, '..'),
  });

  let allIssues = [];

  // 分析每个文件
  componentFiles.forEach(file => {
    const filePath = path.resolve(__dirname, '..', file);
    const colors = extractColorsFromFile(filePath);
    const issues = checkColorContrast(colors, file);

    if (issues.length > 0) {
      allIssues = allIssues.concat(issues);
    }
  });

  // 输出结果
  if (allIssues.length > 0) {
    console.log(`发现 ${allIssues.length} 个颜色对比度问题:\n`);

    allIssues.forEach((issue, index) => {
      console.log(`问题 #${index + 1}`);
      console.log(`  文件: ${issue.file}`);
      console.log(`  行号: ${issue.line}`);
      console.log(
        `  文本颜色: ${issue.textColor}${issue.textClass ? ` (${issue.textClass})` : ''}`
      );
      console.log(
        `  背景颜色: ${issue.bgColor}${issue.bgClass ? ` (${issue.bgClass})` : ''}${issue.default ? ' (默认)' : ''}`
      );
      console.log(`  对比度: ${issue.contrast.toFixed(2)} (WCAG AA要求至少4.5)`);
      console.log(`  建议: ${issue.recommendation}`);
      console.log('');
    });

    // 生成修复建议
    console.log('建议修复方案:');
    allIssues.forEach((issue, index) => {
      const textRgb = parseColor(issue.textColor);
      const bgRgb = parseColor(issue.bgColor);

      if (textRgb && bgRgb) {
        // 如果文本颜色较暗，建议使用更暗的颜色
        if (getLuminance(textRgb) < 0.5) {
          const darkerText = {
            r: Math.max(0, textRgb.r - 50),
            g: Math.max(0, textRgb.g - 50),
            b: Math.max(0, textRgb.b - 50),
          };
          const newContrast = getContrastRatio(darkerText, bgRgb);
          if (newContrast >= MIN_CONTRAST_RATIO_NORMAL_TEXT) {
            console.log(
              `  问题 #${index + 1}: 将文本颜色从 ${issue.textColor} 更改为 ${rgbToHex(darkerText.r, darkerText.g, darkerText.b)} (新对比度: ${newContrast.toFixed(2)})`
            );
          }
        } else {
          // 如果文本颜色较亮，建议使用更亮的颜色
          const lighterText = {
            r: Math.min(255, textRgb.r + 50),
            g: Math.min(255, textRgb.g + 50),
            b: Math.min(255, textRgb.b + 50),
          };
          const newContrast = getContrastRatio(lighterText, bgRgb);
          if (newContrast >= MIN_CONTRAST_RATIO_NORMAL_TEXT) {
            console.log(
              `  问题 #${index + 1}: 将文本颜色从 ${issue.textColor} 更改为 ${rgbToHex(lighterText.r, lighterText.g, lighterText.b)} (新对比度: ${newContrast.toFixed(2)})`
            );
          }
        }
      } else {
        console.log(`  问题 #${index + 1}: 无法自动生成修复建议，请手动检查。`);
      }
    });
  } else {
    console.log('未发现颜色对比度问题，太棒了！');
  }
}

main();
