# Bilingual static portfolio

The English page is `index.html`; the Chinese page is `zh.html`. Both use `styles.css` and the same actual experiment images in `assets/`. Language switching uses ordinary links and works without JavaScript. Repository READMEs also link to these images.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory site
```

Open `http://localhost:8000/` and use the EN / 中文 links. No framework, package installation, account or API key is needed.

## Publish with GitHub Pages

The repository includes `.github/workflows/portfolio-pages.yml`. Before the first deployment, set **Settings → Pages → Build and deployment → Source → GitHub Actions**. Then run **Actions → Deploy portfolio to GitHub Pages → Run workflow** on `main`. Subsequent changes to `site/` on `main` trigger deployment automatically.

Only the `site/` directory is uploaded, not experiment datasets, weights or the rest of the repository. The workflow's deployment environment reports the live URL after success. Do not treat a source commit or a local preview as proof of deployment.

The expected default project-site path is `https://xuzihao723.github.io/xiaomi-auto-drive/`, subject to the repository's Pages settings. Local asset links are relative so they work under the project subpath.

## 后续维护

- 默认英文页：`index.html`；中文页：`zh.html`；顶部点击切换。
- 在两个页面同步修改简介、模块说明和指标，保留评估条件。
- 个人资料只填写已确认的信息；当前仅使用维护者的 GitHub 账号。
- 首次上线时，在 **Settings → Pages** 选择 **GitHub Actions**，然后运行发布工作流；成功后以环境显示的地址为准。
- 图片来自原始实验，发布记录中的周次标签仍保留以便追溯。

Hosting reference: [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
