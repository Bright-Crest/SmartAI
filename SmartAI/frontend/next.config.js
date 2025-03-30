/** @type {import('next').NextConfig} */
const nextConfig = {
  onDemandEntries: {
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
  devIndicators: {
    buildActivity: true,
  },
  env: {
    NEXT_PUBLIC_BACKEND_URL: 'http://localhost:3001',
    NEXT_PUBLIC_FASTAPI_URL: 'http://localhost:8002',
  },
  reactStrictMode: false,
  productionBrowserSourceMaps: true,
  swcMinify: true,
  compiler: {
    styledComponents: true,
  },
  images: {
    domains: ['localhost'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.example.com',
      },
    ],
  },
  async redirects() {
    return [
      {
        source: '/app/:path*',
        destination: '/:path*',
        permanent: true,
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/network-graph.html',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'Content-Security-Policy',
            value: "default-src * 'self' 'unsafe-inline' 'unsafe-eval' data: gap:",
          },
          {
            key: 'Access-Control-Allow-Origin',
            value: '*',
          },
        ],
      },
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
