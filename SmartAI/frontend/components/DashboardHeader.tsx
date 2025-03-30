'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
// 使用自定义图标组件
import { UserIcon, BellIcon, MenuIcon, XIcon, SearchIcon, ChevronDownIcon } from './icons';
import useKeyboardNavigation from '../hooks/useKeyboardNavigation';
import FocusableItem from './a11y/FocusableItem';
import AccessibleMenu from './a11y/AccessibleMenu';
import { useA11y } from '../context/A11yContext';

export default function DashboardHeader() {
  const { t } = useTranslation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  
  const pathname = usePathname();
  
  // 引用菜单容器
  const notificationsRef = useRef<HTMLDivElement>(null);
  const profileMenuRef = useRef<HTMLDivElement>(null);
  const mobileMenuRef = useRef<HTMLDivElement>(null);
  
  // 使用键盘导航Hook
  const { handleEnterAndSpace, manageFocusTrap, saveFocus, restoreFocus } = useKeyboardNavigation();
  const { announce } = useA11y?.() || { announce: () => {} };
  
  // 处理登出点击
  const handleLogout = () => {
    console.log('Logout clicked');
    // 实际的登出逻辑
    announce(t('common.loggingOut'));
  };
  
  // 导航菜单项
  const navigation = [
    { name: t('navigation.dashboard'), href: '/dashboard', current: pathname === '/dashboard' },
    { name: t('navigation.transactions'), href: '/transactions', current: pathname === '/transactions' },
    { name: t('navigation.addresses'), href: '/addresses', current: pathname === '/addresses' },
    { name: t('navigation.networkAnalysis'), href: '/network-analysis', current: pathname === '/network-analysis' },
    { name: t('navigation.alerts'), href: '/alerts', current: pathname === '/alerts' },
    { name: t('navigation.settings'), href: '/settings', current: pathname === '/settings' },
  ];
  
  // 配置个人资料菜单项
  const profileMenuItems = [
    { 
      id: 'profile', 
      label: t('common.profile'), 
      onClick: () => { window.location.href = '/profile'; } 
    },
    { 
      id: 'settings', 
      label: t('common.settings'), 
      onClick: () => { window.location.href = '/settings'; } 
    },
    { 
      id: 'logout', 
      label: t('common.logout'), 
      onClick: handleLogout 
    }
  ];
  
  // 处理通知点击
  const toggleNotifications = () => {
    if (isNotificationsOpen) {
      closeNotifications();
    } else {
      openNotifications();
    }
  };
  
  // 打开通知菜单
  const openNotifications = () => {
    saveFocus();
    setIsNotificationsOpen(true);
    // 关闭其他菜单
    setIsProfileOpen(false);
    setIsMenuOpen(false);
    // 宣告给屏幕阅读器
    announce(t('common.notificationsOpened'));
  };
  
  // 关闭通知菜单
  const closeNotifications = () => {
    setIsNotificationsOpen(false);
    restoreFocus();
    // 宣告给屏幕阅读器
    announce(t('common.notificationsClosed'));
  };
  
  // 处理个人资料点击
  const toggleProfile = () => {
    if (isProfileOpen) {
      closeProfile();
    } else {
      openProfile();
    }
  };
  
  // 打开个人资料菜单
  const openProfile = () => {
    saveFocus();
    setIsProfileOpen(true);
    // 关闭其他菜单
    setIsNotificationsOpen(false);
    setIsMenuOpen(false);
    // 宣告给屏幕阅读器
    announce(t('common.profileMenuOpened'));
  };
  
  // 关闭个人资料菜单
  const closeProfile = () => {
    setIsProfileOpen(false);
    restoreFocus();
    // 宣告给屏幕阅读器
    announce(t('common.profileMenuClosed'));
  };
  
  // 处理菜单点击
  const toggleMenu = () => {
    if (isMenuOpen) {
      closeMenu();
    } else {
      openMenu();
    }
  };
  
  // 打开移动菜单
  const openMenu = () => {
    saveFocus();
    setIsMenuOpen(true);
    // 关闭其他菜单
    setIsNotificationsOpen(false);
    setIsProfileOpen(false);
    // 宣告给屏幕阅读器
    announce(t('common.mainMenuOpened'));
  };
  
  // 关闭移动菜单
  const closeMenu = () => {
    setIsMenuOpen(false);
    restoreFocus();
    // 宣告给屏幕阅读器
    announce(t('common.mainMenuClosed'));
  };
  
  // 监听Escape键关闭菜单
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (isNotificationsOpen) closeNotifications();
        if (isProfileOpen) closeProfile();
        if (isMenuOpen) closeMenu();
      }
    };
    
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isNotificationsOpen, isProfileOpen, isMenuOpen]);
  
  return (
    <header className="bg-white shadow">
      <div className="mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 justify-between">
          <div className="flex">
            <div className="flex flex-shrink-0 items-center">
              <Link href="/dashboard">
                <div className="flex items-center">
                  <Image
                    src="/logo.svg"
                    alt="Chain Intel AI Logo"
                    width={32}
                    height={32}
                    className="h-8 w-auto"
                  />
                  <span className="ml-2 text-xl font-bold text-primary">Chain Intel AI</span>
                </div>
              </Link>
            </div>
            <nav className="hidden sm:ml-6 sm:flex sm:space-x-8" aria-label="主导航">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`inline-flex items-center border-b-2 px-1 pt-1 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
                    item.current
                      ? 'border-primary text-primary'
                      : 'border-transparent text-gray-700 hover:border-gray-300 hover:text-gray-900'
                  }`}
                  aria-current={item.current ? 'page' : undefined}
                >
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
          <div className="hidden sm:ml-6 sm:flex sm:items-center">
            <FocusableItem
              className="rounded-full bg-white p-1 text-gray-700 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
              onClick={toggleNotifications}
              role="button"
              aria-haspopup="true"
              aria-expanded={isNotificationsOpen}
              aria-controls="notifications-menu"
              aria-label={t('common.notifications')}
            >
              <BellIcon className="h-6 w-6" aria-hidden="true" />
            </FocusableItem>

            {/* 通知下拉菜单 */}
            {isNotificationsOpen && (
              <div 
                ref={notificationsRef}
                className="absolute right-10 top-12 z-10 mt-2 w-80 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none"
                role="menu"
                aria-orientation="vertical"
                aria-labelledby="notifications-menu-button"
                id="notifications-menu"
                onKeyDown={manageFocusTrap(notificationsRef, closeNotifications)}
              >
                <div className="border-b border-gray-200 px-4 py-2">
                  <h3 className="text-sm font-medium text-gray-900">{t('common.notifications')}</h3>
                </div>
                <div className="px-4 py-2 text-sm text-gray-700">
                  <p className="mb-2 text-center text-gray-700">{t('common.noNotifications')}</p>
                </div>
                <FocusableItem
                  onClick={closeNotifications}
                  className="text-center text-sm text-primary hover:text-primary-dark px-4 py-2"
                  role="menuitem"
                >
                  {t('common.close')}
                </FocusableItem>
              </div>
            )}

            {/* 使用AccessibleMenu组件实现个人资料菜单 */}
            <div className="relative ml-3">
              <AccessibleMenu
                trigger={
                  <div className="flex rounded-full bg-white text-sm">
                    <span className="sr-only">{t('common.openUserMenu')}</span>
                    <div className="h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">
                      <UserIcon className="h-5 w-5 text-gray-700" aria-hidden="true" />
                    </div>
                  </div>
                }
                items={profileMenuItems}
                placement="bottom"
                menuClassName="w-48"
                onClose={() => setIsProfileOpen(false)}
              />
            </div>
          </div>
          <div className="flex items-center sm:hidden">
            {/* 移动菜单按钮 */}
            <FocusableItem
              className="inline-flex items-center justify-center rounded-md p-2 text-gray-700 hover:bg-gray-100 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary"
              onClick={toggleMenu}
              role="button"
              aria-controls="mobile-menu"
              aria-expanded={isMenuOpen}
              aria-label={isMenuOpen ? t('common.closeMainMenu') : t('common.openMainMenu')}
            >
              {isMenuOpen ? (
                <XIcon className="block h-6 w-6" aria-hidden="true" />
              ) : (
                <MenuIcon className="block h-6 w-6" aria-hidden="true" />
              )}
            </FocusableItem>
          </div>
        </div>
      </div>

      {/* 移动菜单，仅在移动设备上显示 */}
      {isMenuOpen && (
        <div 
          className="sm:hidden" 
          id="mobile-menu"
          ref={mobileMenuRef}
          onKeyDown={manageFocusTrap(mobileMenuRef, closeMenu)}
        >
          <div className="space-y-1 pb-3 pt-2">
            {navigation.map((item, index) => (
              <Link
                key={item.name}
                href={item.href}
                className={`block border-l-4 py-2 pl-3 pr-4 text-base font-medium focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary ${
                  item.current
                    ? 'border-primary bg-primary-50 text-primary'
                    : 'border-transparent text-gray-700 hover:border-gray-300 hover:bg-gray-50 hover:text-gray-900'
                }`}
                aria-current={item.current ? 'page' : undefined}
                tabIndex={0}
              >
                {item.name}
              </Link>
            ))}
          </div>
          <div className="border-t border-gray-200 pb-3 pt-4">
            <div className="flex items-center px-4">
              <div className="flex-shrink-0">
                <div className="h-10 w-10 rounded-full bg-gray-200 flex items-center justify-center">
                  <UserIcon className="h-6 w-6 text-gray-700" aria-hidden="true" />
                </div>
              </div>
              <div className="ml-3">
                <div className="text-base font-medium text-gray-800">{t('common.userProfile')}</div>
                <div className="text-sm font-medium text-gray-500">user@example.com</div>
              </div>
            </div>
            <div className="mt-3 space-y-1">
              <FocusableItem
                onClick={() => { window.location.href = '/profile'; }}
                className="block px-4 py-2 text-base font-medium text-gray-700 hover:bg-gray-50 hover:text-gray-900"
                role="menuitem"
              >
                {t('common.profile')}
              </FocusableItem>
              <FocusableItem
                onClick={() => { window.location.href = '/settings'; }}
                className="block px-4 py-2 text-base font-medium text-gray-700 hover:bg-gray-50 hover:text-gray-900"
                role="menuitem"
              >
                {t('common.settings')}
              </FocusableItem>
              <FocusableItem
                onClick={handleLogout}
                className="block px-4 py-2 text-base font-medium text-gray-700 hover:bg-gray-50 hover:text-gray-900"
                role="menuitem"
              >
                {t('common.logout')}
              </FocusableItem>
            </div>
          </div>
        </div>
      )}
    </header>
  );
} 