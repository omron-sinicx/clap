import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import yaml from '@rollup/plugin-yaml';
import jsyaml from 'js-yaml';
import path from 'path';
import fs from 'fs';

const data = jsyaml.load(
  fs.readFileSync(path.resolve(__dirname, 'template.yaml'), 'utf8')
);

// Resolve the active theme at build time so its SCSS can be imported
// statically (Vite extracts it into a <link> in <head>, avoiding FOUC).
const activeTheme = path.resolve(
  __dirname,
  'src/scss',
  data.theme === 'dark' ? 'dark-theme.scss' : 'theme.scss'
);

// Bake OGP/Twitter meta into <head> at build time (and in dev) from
// template.yaml, so social crawlers see them without running JS. This
// replaces the react-snap prerender that previously did this.
const ogpPlugin = () => ({
  name: 'inject-ogp',
  transformIndexHtml: () => [
    { tag: 'title', children: data.title, injectTo: 'head' },
    ...[
      { name: 'description', content: data.description },
      { property: 'og:site_name', content: data.organization },
      { property: 'og:type', content: 'article' },
      { property: 'og:title', content: data.title },
      { property: 'og:description', content: data.description },
      { property: 'og:image', content: data.image },
      { property: 'og:image:alt', content: data.description },
      { property: 'og:image:width', content: '1200' },
      { property: 'og:image:height', content: '600' },
      { property: 'og:url', content: data.url },
      { name: 'twitter:card', content: 'summary_large_image' },
      { name: 'twitter:title', content: data.title },
      { name: 'twitter:image:src', content: data.image },
      { name: 'twitter:description', content: data.description },
      { name: 'twitter:url', content: data.url },
      { name: 'twitter:site', content: data.twitter },
    ].map((attrs) => ({ tag: 'meta', attrs, injectTo: 'head' })),
  ],
});

// https://vite.dev/config/
export default defineConfig({
  base: './',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@active-theme': activeTheme,
    },
  },
  plugins: [react(), yaml(), ogpPlugin()],
  build: {
    outDir: 'build',
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
      },
    },
    target: 'es2015',
  },
  server: {
    host: '0.0.0.0',
    port: 8080,
  },
  css: {
    preprocessorOptions: {
      scss: {
        silenceDeprecations: ['import'],
        quietDeps: true,
      },
    },
  },
});
