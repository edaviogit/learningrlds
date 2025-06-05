# Jupyter Lab 中显示 Mermaid 图表的解决方案

## 问题描述
Jupyter Lab 默认不支持 Mermaid 图表渲染，需要额外配置。

## 解决方案

### 🚀 方案1: 安装 JupyterLab-Mermaid 扩展 (推荐)

```bash
# 安装 mermaid 扩展
pip install jupyterlab-mermaid

# 如果使用 JupyterLab 3.x
jupyter labextension install @jupyterlab/javascript-extension
jupyter labextension install jupyterlab-mermaid

# 如果使用 JupyterLab 4.x (最新版本)
pip install jupyterlab_mermaid
```

安装后重启 Jupyter Lab：
```bash
jupyter lab
```

### 📱 方案2: 使用在线工具查看

1. **Mermaid Live Editor**: https://mermaid.live/
   - 复制 notebook 中的 mermaid 代码
   - 粘贴到在线编辑器查看

2. **GitHub Gist**: 
   - 创建包含 mermaid 代码的 .md 文件
   - GitHub 会自动渲染 mermaid 图表

### 🔧 方案3: 转换为图片格式

使用 mermaid-cli 工具：
```bash
# 安装 mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# 转换 mermaid 为图片
mmdc -i diagram.mmd -o diagram.png
```

### 🐍 方案4: 使用 Python 可视化库替代

将 Mermaid 图表转换为 Python 可视化代码。

### 📝 方案5: 修改 Notebook 内容

在 notebook 中添加图片版本的图表。 