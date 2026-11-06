import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Literal loopback addresses: "localhost" may resolve to ::1, where another user
    // could listen and receive the page's token or the proxied requests carrying it.
    host: '127.0.0.1',
    port: 5173,
    // Fail rather than move to another port: the docs send the token to this one.
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8200',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8200',
        ws: true,
      },
    },
  },
})
