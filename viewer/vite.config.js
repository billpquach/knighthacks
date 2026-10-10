import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    proxy: Object.fromEntries(['/state', '/frame.jpg', '/frame-raw.jpg', '/viewer-info', '/scene', '/metrics'].map(path => [path, 'http://127.0.0.1:8080'])),
  },
  build: { rollupOptions: { output: { manualChunks: id => id.includes('/node_modules/three/') ? 'three' : undefined } } },
});
