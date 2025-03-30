/**
 * Chakra UI 主题配置
 */
import { extendTheme } from '@chakra-ui/react';

// 使用extendTheme创建主题
const theme = extendTheme({
  colors: {
    brand: {
      50: '#E6F6FF',
      100: '#B3E0FF',
      200: '#80CBFF',
      300: '#4DB5FF',
      400: '#1A9FFF',
      500: '#0080E6',
      600: '#0066B3',
      700: '#004D80',
      800: '#00334D',
      900: '#001A26',
    },
  },
  fonts: {
    heading: 'Inter, system-ui, sans-serif',
    body: 'Inter, system-ui, sans-serif',
  },
  config: {
    initialColorMode: 'light',
    useSystemColorMode: false,
  },
});

export default theme;
