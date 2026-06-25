import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The proxy target must be a full URL with a scheme. Railway's RAILWAY_*_DOMAIN
// variables expose a bare host (no scheme), which makes Vite's proxy crash with
// "Cannot read properties of null (reading 'split')". Normalise to a full URL.
function resolveApiTarget() {
  const raw = process.env.VITE_API_URL
  if (!raw) return 'http://localhost:8000'
  if (/^https?:\/\//.test(raw)) return raw
  // Internal Railway hosts and localhost speak plain HTTP; public hosts use HTTPS.
  const scheme = /\.railway\.internal|^localhost|^127\.0\.0\.1/.test(raw) ? 'http' : 'https'
  return `${scheme}://${raw}`
}

export default defineConfig(({ command }) => ({
  plugins: [react()],
  esbuild: {
    // Strip console.* and debugger statements from production builds.
    drop: command === 'build' ? ['console', 'debugger'] : [],
  },
  server: {
    port: 5173,
    host: true,
    // The dev server runs behind Railway's public host, which Vite blocks by
    // default. Allow any Railway subdomain; add custom domains to this list.
    allowedHosts: ['.up.railway.app'],
    proxy: {
      '/api': {
        target: resolveApiTarget(),
        changeOrigin: true,
      }
    }
  },
}))
