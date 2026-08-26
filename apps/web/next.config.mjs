/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The API base is read at request time on the server, so the same build can
  // be promoted between environments without rebuilding.
  env: { API_INTERNAL_URL: process.env.API_INTERNAL_URL ?? 'http://localhost:8000' },
};
export default nextConfig;
