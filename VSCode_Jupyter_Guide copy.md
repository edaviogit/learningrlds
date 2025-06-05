# VSCode 中使用 RLDS Jupyter Notebook 完整指南

## 🎯 问题解决

您的问题已完全解决！我已经为VSCode环境专门优化了notebook。

## 📁 推荐使用的文件

**使用这个文件**: `RLDS_tutorial_vscode.ipynb`
- ✅ 针对VSCode优化
- ✅ 包含Mermaid图表说明
- ✅ 提供多种查看方案
- ✅ 6个Mermaid图表已添加文字说明

## 🚀 立即开始的步骤

### 1. 安装VSCode Mermaid扩展 (推荐)

```
步骤：
1. 在VSCode中按 Ctrl+Shift+X (或 Cmd+Shift+X on Mac)
2. 搜索 "Mermaid Preview"
3. 安装 "Mermaid Preview" 扩展 (作者: bierner)
   - 扩展ID: bierner.markdown-mermaid
4. 重启VSCode
```

### 2. 打开优化版notebook

```bash
# 在VSCode中打开
code RLDS_tutorial_vscode.ipynb
```

### 3. 查看Mermaid图表的三种方法

现在您有3种方式查看图表：

#### 方法1: 直接在VSCode中查看 (最佳)
- 安装扩展后，Mermaid图表会直接在markdown单元格中渲染

#### 方法2: 在线预览 (临时方案)
- 复制mermaid代码到 https://mermaid.live/
- 即时在线查看图表

#### 方法3: 文字版说明 (备用方案)
- 每个图表下方都有详细的文字版架构说明

## 🔧 VSCode配置优化 (可选)

在VSCode设置中添加以下配置：

### 打开设置
```
1. 按 Ctrl+, (或 Cmd+, on Mac) 打开设置
2. 点击右上角的"打开设置(JSON)"图标
3. 添加以下配置
```

### 配置内容
```json
{
    // Mermaid相关设置
    "markdown.mermaid.theme": "default",
    "markdown.preview.breaks": true,
    
    // Jupyter相关设置  
    "jupyter.enableCellCodeLens": true,
    "jupyter.sendSelectionToInteractiveWindow": false,
    "jupyter.interactiveWindowMode": "perFile",
    
    // 编辑器设置
    "editor.wordWrap": "on",
    "editor.fontSize": 14
}
```

## 📊 文件对比

| 文件名 | 适用环境 | Mermaid支持 | 推荐度 |
|--------|----------|-------------|--------|
| `RLDS_tutorial.ipynb` | 独立Jupyter Lab | 需要扩展 | ⭐⭐⭐ |
| `RLDS_tutorial_vscode.ipynb` | **VSCode** | **优化支持** | **⭐⭐⭐⭐⭐** |

## ✅ 验证效果

安装扩展并重启VSCode后，您应该看到：

1. **设置指南单元格** - 在notebook开头的详细说明
2. **图表说明单元格** - 每个Mermaid图表前的使用提示  
3. **文字版架构说明** - 每个图表的文字描述
4. **原始Mermaid代码** - 可直接渲染或复制使用

## 🎯 测试步骤

1. 打开 `RLDS_tutorial_vscode.ipynb`
2. 查看第二个单元格的设置指南
3. 滚动到任意Mermaid图表位置
4. 确认能看到：
   - 📊 架构图说明
   - 📝 文字版架构说明  
   - 🔧 Mermaid代码块

## 🔍 故障排除

### 如果扩展安装后仍看不到图表：

1. **检查扩展**：确认"Mermaid Preview"已启用
2. **重启VSCode**：完全关闭后重新打开
3. **检查语法**：确保mermaid代码块格式正确
4. **使用备用方案**：直接查看文字版说明

### 常见问题：

**Q: 为什么有些图表显示不完整？**
A: 复杂的Mermaid图表可能需要调整VSCode窗口大小，或使用在线预览。

**Q: 可以自定义图表主题吗？**  
A: 可以在settings.json中修改 `"markdown.mermaid.theme"` 的值。

## 🎉 总结

现在您已经有了完美的VSCode RLDS学习环境：

- ✅ **专门优化的notebook**: `RLDS_tutorial_vscode.ipynb`
- ✅ **多种图表查看方案**: 扩展渲染、在线预览、文字说明
- ✅ **详细的使用指南**: 每个图表都有说明
- ✅ **VSCode原生支持**: 无需额外安装Python库

立即开始学习RLDS吧！🚀 