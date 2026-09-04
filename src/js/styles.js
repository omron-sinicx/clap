import 'katex/dist/katex.min.css';
// Static import so Vite extracts the theme CSS into a render-blocking
// <link> in <head> (no FOUC). The active theme is resolved at build time
// via the '@active-theme' alias in vite.config.js from template.yaml.
import '@active-theme';

import UIkit from 'uikit';
import Icons from 'uikit/dist/js/uikit-icons';

UIkit.use(Icons);
