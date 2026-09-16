/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Traced output: only the files actually reached are shipped, which is a
  // fraction of node_modules and what the runtime image copies.
  output: 'standalone',
  // The workspace root, so tracing follows into packages/.
  outputFileTracingRoot: new URL('../../', import.meta.url).pathname,
  // The API base is read at request time on the server, so the same build can
  // be promoted between environments without rebuilding.
  env: { API_INTERNAL_URL: process.env.API_INTERNAL_URL ?? 'http://localhost:8000' },
};
export default nextConfig;
