import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  // Served from https://rabrooks.github.io/inferplan/ — remove if a custom domain is added
  base: '/inferplan/',
  plugins: [react(), tailwindcss()],
})
