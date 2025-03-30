import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useLazyImage } from '../hooks/useIntersectionObserver';

interface LazyImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  className?: string;
  placeholderColor?: string;
}

/**
 * 懒加载图片组件
 * 当图片进入视口时才加载图片
 */
const LazyImage: React.FC<LazyImageProps> = ({
  src,
  alt,
  width,
  height,
  className = '',
  placeholderColor = '#f3f4f6'
}) => {
  // 使用自定义的ref和状态管理
  const imgRef = useRef<HTMLImageElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [imageSrc, setImageSrc] = useState('');
  
  // 创建自己的Intersection Observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          const img = new Image();
          img.src = src;
          img.onload = () => {
            setImageSrc(src);
            setIsLoaded(true);
          };
          // 一旦加载，停止观察
          if (imgRef.current) observer.unobserve(imgRef.current);
        }
      },
      { rootMargin: '200px' }
    );
    
    if (imgRef.current) {
      observer.observe(imgRef.current);
    }
    
    return () => {
      if (imgRef.current) observer.unobserve(imgRef.current);
    };
  }, [src]);
  
  return (
    <div 
      className={`relative overflow-hidden ${className}`}
      style={{ width, height, backgroundColor: placeholderColor }}
    >
      {/* 占位符 */}
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
        </div>
      )}
      
      {/* 图片 */}
      <motion.img
        ref={imgRef}
        src={imageSrc || ''}
        alt={alt}
        width={width}
        height={height}
        className={`w-full h-full object-cover transition-opacity duration-300 ${isLoaded ? 'opacity-100' : 'opacity-0'}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: isLoaded ? 1 : 0 }}
        transition={{ duration: 0.3 }}
      />
    </div>
  );
};

// 使用React.memo优化渲染性能
export default React.memo(LazyImage); 