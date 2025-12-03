# SAEMark Project Page

This folder contains the static website for the SAEMark project.

## Structure

```
pages/
├── index.html          # Main project page
├── assets/
│   ├── css/
│   │   └── style.css   # Styling
│   ├── js/
│   │   └── main.js     # Interactivity (copy button, animations)
│   └── images/         # Figures from paper (if needed)
└── README.md           # This file
```

## Deployment

The site is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the `main` branch.

**Live URL**: https://zhuohaoyu.github.io/SAEMark

## Local Preview

To preview locally, you can use any static server:

```bash
# Using Python
cd pages && python -m http.server 8000

# Using Node.js (npx)
npx serve pages

# Or simply open pages/index.html in a browser
```

## Links

- **Paper**: https://arxiv.org/abs/2508.08211
- **Code**: https://github.com/zhuohaoyu/SAEMark

