import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import AddressFormatter from '../app/components/AddressFormatter';
import RiskBadge from './RiskBadge';
import { useA11y } from '../context/A11yContext';
import FocusableItem from './a11y/FocusableItem';

const SimilarAddresses = ({ address, similarAddresses = [], isLoading = false }) => {
  const { t } = useTranslation();
  const [showAll, setShowAll] = useState(false);
  const { announce } = useA11y();
  
  // 如果没有相似地址数据，显示空状态
  if (!isLoading && (!similarAddresses || similarAddresses.length === 0)) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          {t('address.similar.title')}
        </h2>
        <div className="bg-gray-50 rounded-md p-6 text-center">
          <p className="text-gray-700">{t('address.similar.noData')}</p>
        </div>
      </div>
    );
  }
  
  // 显示加载状态
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          {t('address.similar.title')}
        </h2>
        <div className="animate-pulse space-y-4" aria-label={t('loading.state')}>
          {[...Array(3)].map((_, i) => (
            <div key={i} className="flex items-center space-x-4">
              <div className="h-12 w-12 rounded-full bg-gray-200"></div>
              <div className="flex-1 space-y-2">
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  // 决定显示多少地址
  const displayAddresses = showAll 
    ? similarAddresses 
    : similarAddresses.slice(0, 5);
  
  const handleShowMoreLess = () => {
    setShowAll(!showAll);
    announce(
      showAll
        ? t('address.similar.collapsed')
        : t('address.similar.expanded', { count: similarAddresses.length })
    );
  };

  const handleViewAddress = (address) => {
    window.open(`/addresses?address=${address}`, '_blank');
  };
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-8">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800">
          {t('address.similar.title')}
        </h2>
        <div className="text-sm">
          <span className="bg-blue-200 text-blue-900 text-xs font-medium px-2.5 py-0.5 rounded">
            AI {t('address.similar.powered')}
          </span>
        </div>
      </div>
      
      <p className="text-sm text-gray-700 mb-6">
        {t('address.similar.description')}
      </p>
      
      <div className="space-y-4" role="list" aria-label={t('address.similar.list')}>
        {displayAddresses.map((item, index) => (
          <div 
            key={index} 
            className="flex flex-col sm:flex-row sm:items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            role="listitem"
          >
            <div className="flex items-center mb-2 sm:mb-0">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-200 flex items-center justify-center mr-3">
                <span className="text-blue-900 font-medium">{item.similarityScore}%</span>
              </div>
              <div>
                <div className="font-mono text-sm text-gray-900">
                  <AddressFormatter address={item.address} />
                </div>
                <div className="text-xs text-gray-700 mt-1">
                  {item.tags.map((tag, i) => (
                    <span key={i} className="mr-2">#{tag}</span>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <RiskBadge level={item.riskLevel} />
              <FocusableItem 
                className="text-blue-800 hover:text-blue-900 text-sm font-medium"
                onClick={() => handleViewAddress(item.address)}
                role="button"
                aria-label={t('address.similar.viewAddress', { address: item.address })}
              >
                {t('address.similar.view')}
              </FocusableItem>
            </div>
          </div>
        ))}
      </div>
      
      {similarAddresses.length > 5 && (
        <div className="mt-4 text-center">
          <FocusableItem
            onClick={handleShowMoreLess}
            className="text-blue-800 hover:text-blue-900 text-sm font-medium"
            role="button"
            aria-expanded={showAll}
          >
            {showAll 
              ? t('address.similar.showLess') 
              : t('address.similar.showMore', { count: similarAddresses.length - 5 })}
          </FocusableItem>
        </div>
      )}
    </div>
  );
};

export default SimilarAddresses; 