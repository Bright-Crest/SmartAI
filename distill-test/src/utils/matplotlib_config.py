"""
提供matplotlib配置工具，特别是中文字体支持
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import os
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import logger

def setup_chinese_fonts():
    """配置matplotlib支持中文显示"""
    # 尝试多种中文字体，以适应不同的操作系统
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 
        'KaiTi', 'NSimSun', 'FangSong', 'WenQuanYi Zen Hei', 
        'AR PL UMing CN', 'PingFang SC', 'Source Han Sans CN', 
        'Noto Sans CJK SC'
    ]
    
    # 查找可用的中文字体
    found_font = None
    for font in chinese_fonts:
        try:
            font_path = font_manager.findfont(font_manager.FontProperties(family=font))
            if font_path and font_path.strip():
                found_font = font
                break
        except:
            continue
    
    # 设置字体
    if found_font:
        logger.info(f"使用中文字体: {found_font}")
        plt.rcParams['font.family'] = found_font
    else:
        # 使用sans-serif字体族
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial Unicode MS'] + plt.rcParams['font.sans-serif']
        logger.info(f"使用sans-serif字体族，添加中文字体支持")
    
    # 修复负号显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 根据操作系统查找额外的中文字体
    system = platform.system()
    if system == 'Windows':
        # Windows系统通常包含SimHei和Microsoft YaHei字体
        try:
            # 重新扫描字体
            font_manager._rebuild()
        except:
            pass
    elif system == 'Linux':
        # 在Linux上查找中文字体
        try:
            # 重新加载字体缓存
            font_dirs = [
                '/usr/share/fonts/truetype/wqy',
                '/usr/share/fonts'
            ]
            for font_dir in font_dirs:
                if os.path.exists(os.path.join(font_dir, 'wqy-zenhei.ttc')):
                    font_manager.fontManager.addfont(os.path.join(font_dir, 'wqy-zenhei.ttc'))
                    break
        except Exception as e:
            logger.warning(f"加载Linux中文字体失败: {str(e)}")
    elif system == 'Darwin':  # macOS
        # 在macOS上查找中文字体
        try:
            # macOS上的一些常见中文字体路径
            font_dirs = [
                '/System/Library/Fonts',
                '/Library/Fonts',
                os.path.expanduser('~/Library/Fonts')
            ]
            for font_dir in font_dirs:
                if os.path.exists(os.path.join(font_dir, 'PingFang.ttc')):
                    font_manager.fontManager.addfont(os.path.join(font_dir, 'PingFang.ttc'))
                    break
        except Exception as e:
            logger.warning(f"加载macOS中文字体失败: {str(e)}")
    
    # 强制Matplotlib使用所选字体
    font_properties = font_manager.FontProperties(family=plt.rcParams['font.family'])
    mpl.rcParams['font.family'] = font_properties.get_family()[0]
    
    return True

def get_chinese_font():
    """获取中文字体属性对象
    
    Returns:
        FontProperties: 中文字体属性对象
    """
    # 尝试多种中文字体，以适应不同的操作系统
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 
        'KaiTi', 'NSimSun', 'FangSong', 'WenQuanYi Zen Hei', 
        'AR PL UMing CN', 'PingFang SC', 'Source Han Sans CN', 
        'Noto Sans CJK SC'
    ]
    
    # 尝试创建一个字体属性对象
    font_prop = None
    
    # 1. 先尝试从当前matplotlib配置中获取字体
    if plt.rcParams['font.family'] != 'sans-serif' and plt.rcParams['font.family'] != ['sans-serif']:
        font_prop = FontProperties(family=plt.rcParams['font.family'])
        logger.debug(f"使用matplotlib配置的字体: {plt.rcParams['font.family']}")
        return font_prop
    
    # 2. 尝试找到系统中可用的中文字体
    for font in chinese_fonts:
        try:
            font_path = font_manager.findfont(font_manager.FontProperties(family=font))
            if font_path and font_path.strip():
                font_prop = FontProperties(family=font)
                logger.debug(f"找到可用的中文字体: {font}")
                return font_prop
        except:
            continue
    
    # 3. 如果找不到任何中文字体，尝试使用系统默认字体
    try:
        # 使用sans-serif族中的第一个字体
        if plt.rcParams['font.sans-serif']:
            font_prop = FontProperties(family=plt.rcParams['font.sans-serif'][0])
            logger.debug(f"使用sans-serif族中的字体: {plt.rcParams['font.sans-serif'][0]}")
            return font_prop
    except:
        pass
    
    # 4. 如果以上方法都失败，返回一个默认的FontProperties对象
    logger.warning("未能找到合适的中文字体，使用默认字体")
    return FontProperties()

# 在模块导入时自动配置
setup_chinese_fonts() 