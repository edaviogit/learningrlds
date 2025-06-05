# VSCode 中 Jupyter Notebook 显示 Mermaid 图表解决方案

## VSCode Jupyter 环境特点

VSCode 使用内置的 Jupyter 扩展，与独立的 JupyterLab 有不同的渲染机制。

## 推荐解决方案

### 🥇 方案1: 安装 Mermaid 预览扩展 (最佳)

在 VSCode 中安装以下扩展：

1. **Mermaid Preview** 扩展
   - 扩展ID: `bierner.markdown-mermaid`
   - 可以在 markdown 单元格中预览 mermaid 图表

2. **Markdown Preview Mermaid Support** 扩展
   - 扩展ID: `matt-meyers.vscode-mermaid-preview`

### 安装步骤：
```
1. 按 Ctrl+Shift+X 打开扩展面板
2. 搜索 "mermaid preview"
3. 安装 "Mermaid Preview" 扩展
4. 重启 VSCode
```

### 🚀 方案2: 使用 Python Mermaid 库

在 notebook 中直接使用 Python 库渲染：

```python
# 安装 python-mermaid 库
!pip install mermaid

# 或使用 pyvis、graphviz 等可视化库
!pip install pyvis graphviz
```

### 📱 方案3: 在线预览 (临时方案)

使用 **Mermaid Live Editor**: https://mermaid.live/
- 复制 notebook 中的 mermaid 代码
- 粘贴到在线编辑器即时查看

### 🔧 方案4: 转换为 Python 可视化代码

将 Mermaid 图表重写为 Python 代码，使用：
- `matplotlib` + `networkx` (网络图)
- `plotly` (交互式图表) 
- `graphviz` (流程图)

### 📝 方案5: 修改 Notebook 显示方式

在 markdown 单元格中添加提示文本：

```markdown
> **注意**: 此处为 Mermaid 图表，请：
> 1. 安装 VSCode Mermaid Preview 扩展查看
> 2. 或访问 https://mermaid.live/ 在线预览
> 3. 图表代码如下：
```

## VSCode 特定配置

### settings.json 配置
```json
{
    "markdown.mermaid.theme": "default",
    "markdown.preview.breaks": true,
    "jupyter.enableCellCodeLens": true
}
```

### 重启 VSCode 后的验证

在 markdown 单元格中测试：
```mermaid
graph TD
    A[开始] --> B[安装扩展]
    B --> C[重启VSCode]
    C --> D[查看效果]
``` 