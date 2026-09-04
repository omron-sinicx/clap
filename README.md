# 🎨 sinicx-template

[![build](https://github.com/omron-sinicx/projectpage-template/actions/workflows/build.yaml/badge.svg)](https://github.com/omron-sinicx/projectpage-template/actions/workflows/build.yaml) [![build](https://github.com/omron-sinicx/projectpage-template/actions/workflows/lint.yaml/badge.svg)](https://github.com/omron-sinicx/projectpage-template/actions/workflows/lint.yaml)

- A project page template built with ⚛️ [React](https://ja.reactjs.org/) + 🎨 [UIKit](https://getuikit.com/)
- **Demo**: ⛅[light-theme](https://omron-sinicx.github.io/mabr/) / [src](https://github.com/omron-sinicx/mabr/tree/project-page) 🕶️ [dark-theme](https://omron-sinicx.github.io/maru/) / [src](https://github.com/omron-sinicx/mabr/tree/project-page)

> [!TIP]
> You can switch themes by setting [theme field in template.yaml](https://github.com/omron-sinicx/projectpage-template/blob/main/template.yaml#L1-L2)

```yaml
theme: default # default || dark
```

## 🚀 Getting Started

### 📋 Prerequisites | 🪟WSL 🐧Linux 🍎MacOS

#### 🔧 Node.js Installation

We recommend using [Volta](https://volta.sh/) for Node.js version management:

```bash
curl https://get.volta.sh/ | bash
```

```bash
# Restart your shell (The configurations is automatically added to your *shrc || *shenv)
# export VOLTA_HOME="$HOME/.volta"
# export PATH="$VOLTA_HOME/bin:$PATH"

volta install node@20.11.0
volta pin node@20.11.0
```

Verify installation:

```bash
node --version  # Should show v20.11.0
npm --version   # Should show 10.2.4+
```

## 🛠️ Development

### 📥 Installation

```sh
npm install
```

### 💻 Development Mode

```bash
npm run dev
```

### 🏗️ Production Mode

```bash
npm run clean
npm run build
npm run preview
```

#### 🔎 OGP / static `<head>`

OGP and Twitter card meta tags are baked into `<head>` at build time from
`template.yaml` by a small Vite plugin (`ogpPlugin` in `vite.config.js`), so
social crawlers see them without running JavaScript — no headless browser
(Puppeteer / react-snap) required. The theme stylesheet is also linked
render-blocking in `<head>` to avoid a flash of unstyled content (FOUC).

You can verify the baked-in meta after `npm run build`:

```bash
grep -E 'og:|twitter:|<title>' build/index.html
```

### 📋 Template

Complete `template.yaml` by filling in the required values. Use null for any unavailable content (e.g., `blog: null`).

```yaml
organization: OMRON SINIC X
twitter: "@omron_sinicx"
title: Path Planning using Neural A* Search
conference: ICML2021
resources:
  paper: https://arxiv.org/abs/1909.13111
  code: https://github.com/omron-sinicx/multipolar
  video: https://www.youtube.com/embed/adUnIj83RtU
  blog: https://medium.com/sinicx/multipolar-multi-source-policy-aggregation-for-transfer-reinforcement-learning-between-diverse-bc42a152b0f5
  ...
```

## 🎨 Customization

### 🔧 Styling

- Customize appearance by modifying UIKit variables in `src/scss/theme.scss` (zero hand-written CSS)
- Extend `*.jsx` files with components from:
  - 🎨 [UIKit Components](https://getuikit.com/docs/introduction)
  - 🎯 [React-Icons](https://react-icons.github.io/react-icons/)

### 📁 Project Structure

```
template.yaml       # Configuration
src/
├── components/     # React components
├── html/           # HTML templates
├── media/          # Media assets (relocated to assets/ automatically)
├── videos/         # Video content
├── js/             # JavaScript files
├── pages/          # Page templates
└── scss/           # Styling
```

## 🚀 Release your project page automatically by GitHub Actions

- example project: https://github.com/omron-sinicx/mabr/tree/project-page

### :octocat: Deploy from GitHub Actions

- Navigate to `https://github.com/{your-github-repo-path}/settings/pages`
- Select **GitHub Actions** at Build and Deployment > Source
- See also: [GitHub Documentation](https://docs.github.com/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) and [actions/deploy-pages](https://github.com/actions/deploy-pages)

![image](https://github.com/user-attachments/assets/4f1ad0f3-46f8-4ab0-8a0c-062d2fba7b46)

> [!NOTE]
> When using GitHub Actions to deploy a site on GitHub Pages, the source code is built internally during the workflow run. Only the build artifacts (e.g., HTML, CSS, JS) are deployed to the GitHub Pages environment, while the repository itself retains only the source code.

### 🌿 Push project page source to "project-page" branch

- `$ git remote add github {your-github-repo-path}`
- `$ git push github {local-project-page-branch}:project-page`
- See also: https://github.com/omron-sinicx/projectpage-template/blob/main/.github/workflows/deploy.yaml

### TroubleShooting

<details>
<summary>Branch "project-page" is not allowed to deploy to github-pages due to environment protection rules</summary>
Navigate to Settings > Environments > github-pages > 🗑️

![image](https://github.com/user-attachments/assets/ddaa751d-cedc-4665-86a1-8afd88e04e52)

</details>

## 🔍 SEO & Social Sharing

### 🌐 OGP Support

- OGP meta tags are generated from `template.yaml` and baked into the static `<head>` at build time by the Vite `ogpPlugin`, so they are present in `build/index.html` for both local builds and GitHub Actions deploys (no headless browser required).
- Example: [Twitter Card Preview](https://x.com/omron_sinicx/status/1847150071143715312)

## 🐶 Husky & Typo Checking

### Automatic Typo Checking

This project uses Husky to automatically check for typos during commits. The check is performed using the `typos` tool and only runs on staged files (files that have been `git add`ed).

### Manual Typo Checking

To manually check for typos without committing:

```bash
npx typos
```

This command checks all files in the project, regardless of whether they are staged for commit or not.

### Disabling Git Hooks

To temporarily disable all git hooks (including lint, format, and typo validation) during commit:

```bash
export HUSKY=0
```

You can re-enable hooks by unsetting the variable or starting a new terminal session.

## 🤝 Contributing

Issues and PRs welcome! Feel free to [open an issue](https://github.com/omron-sinicx/projectpage-template/issues)
