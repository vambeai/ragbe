import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

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
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
}))
