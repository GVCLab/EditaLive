import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
export default defineConfig({ plugins: [svelte()], server: { proxy: { '/api': { target: 'http://localhost:7860', ws: true } } } });
