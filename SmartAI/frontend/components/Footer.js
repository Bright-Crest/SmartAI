import React from 'react';
import Link from 'next/link';
import { FaGithub, FaTwitter, FaDiscord, FaEnvelope } from 'react-icons/fa';

/**
 * 页脚组件
 *
 * @returns {JSX.Element} 页脚组件
 */
export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white border-t border-gray-200 dark:bg-gray-800 dark:border-gray-700">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 md:flex md:items-center md:justify-between lg:px-8">
        <div className="flex justify-center space-x-6 md:order-2">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
          >
            <span className="sr-only">GitHub</span>
            <FaGithub className="h-5 w-5" />
          </a>
          <a
            href="https://twitter.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
          >
            <span className="sr-only">Twitter</span>
            <FaTwitter className="h-5 w-5" />
          </a>
          <a
            href="https://discord.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
          >
            <span className="sr-only">Discord</span>
            <FaDiscord className="h-5 w-5" />
          </a>
          <a
            href="mailto:contact@chainintelai.com"
            className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
          >
            <span className="sr-only">邮箱</span>
            <FaEnvelope className="h-5 w-5" />
          </a>
        </div>
        <div className="mt-4 md:mt-0 md:order-1 flex flex-col md:flex-row md:items-center">
          <div>
            <p className="text-center text-base text-gray-500 dark:text-gray-400">
              &copy; {currentYear} ChainIntelAI. 保留所有权利。
            </p>
          </div>
          <div className="mt-2 md:mt-0 md:ml-6 flex justify-center space-x-4">
            <Link
              href="/privacy"
              className="text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300"
            >
              隐私政策
            </Link>
            <Link
              href="/terms"
              className="text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300"
            >
              服务条款
            </Link>
            <Link
              href="/about"
              className="text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300"
            >
              关于我们
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
