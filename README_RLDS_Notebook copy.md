# RLDS Jupyter Notebook 使用说明

## 概述

这个仓库包含了从 `RLDS_explained.md` 文档转换生成的交互式 Jupyter Notebook 教程。

## 生成的文件

- `RLDS_tutorial.ipynb` - 主要的教程notebook (推荐使用)
- `RLDS_explained.ipynb` - 基础版本的notebook
- `convert_md_to_ipynb_improved.py` - 改进版的转换脚本

## 安装依赖

在运行notebook之前，请安装必要的依赖：

```bash
# 基础依赖
pip install tensorflow tensorflow-datasets numpy matplotlib

# RLDS 相关依赖 (可选)
pip install rlds dm-env envlogger

# Jupyter 环境
pip install jupyter jupyterlab

# 如果使用conda
conda install tensorflow tensorflow-datasets numpy matplotlib jupyter
```

## 使用方法

### 1. 启动 Jupyter

```bash
# 使用 Jupyter Notebook
jupyter notebook

# 或使用 JupyterLab (推荐)
jupyter lab
```

### 2. 打开教程

在Jupyter界面中打开 `RLDS_tutorial.ipynb`

### 3. 运行教程

按顺序运行每个单元格：
- **Shift + Enter**: 运行当前单元格并移到下一个
- **Ctrl + Enter**: 运行当前单元格但停留在当前位置
- **Alt + Enter**: 运行当前单元格并在下方插入新单元格

## Notebook 结构

生成的notebook包含以下类型的单元格：

### Markdown 单元格 (47个)
- 章节标题和说明
- 概念解释
- 表格和数据结构说明
- Mermaid图表 (在某些环境中可能需要插件支持)

### Code 单元格 (33个)
- Python代码示例
- RLDS数据处理示例
- 数据集构建代码
- 分析和可视化代码

## 主要内容

1. **RLDS基础概念**
   - 架构图和数据结构
   - Episode和Step的定义

2. **数据集构建**
   - 自定义数据集创建
   - TFDS集成

3. **数据处理**
   - 数据变换和优化
   - 批处理和性能优化

4. **实际应用案例**
   - 离线强化学习
   - 模仿学习
   - 数据集分析

5. **高级功能**
   - 多智能体支持
   - 并行处理
   - 性能优化

## 注意事项

### Mermaid 图表支持

部分单元格包含Mermaid图表。要正确显示这些图表，您可能需要：

1. **使用JupyterLab + 插件**:
   ```bash
   pip install jupyterlab-mermaid
   jupyter labextension install jupyterlab-mermaid
   ```

2. **在线Mermaid编辑器**: 
   复制Mermaid代码到 [mermaid.live](https://mermaid.live/) 查看

3. **替代方案**: 
   图表内容也以文字形式在markdown中进行了说明

### 数据集下载

某些代码示例可能涉及大型数据集下载。请确保：
- 有足够的磁盘空间
- 网络连接稳定
- 了解数据使用政策

### GPU支持

代码包含了GPU配置，但如果没有GPU也可以正常运行：
```python
# 会自动检测并配置GPU，无GPU时自动使用CPU
tf.config.experimental.set_memory_growth(...)
```

## 故障排除

### 常见问题

1. **ImportError: No module named 'rlds'**
   ```bash
   pip install rlds
   # 或者
   pip install git+https://github.com/deepmind/rlds.git
   ```

2. **TensorFlow版本不兼容**
   ```bash
   pip install --upgrade tensorflow>=2.8.0
   ```

3. **内存不足**
   - 重启Jupyter kernel
   - 减少批处理大小
   - 使用数据集的较小子集

### 获取帮助

- 查看原始文档: `RLDS_explained.md`
- RLDS官方文档: [https://github.com/deepmind/rlds](https://github.com/deepmind/rlds)
- TensorFlow Datasets: [https://www.tensorflow.org/datasets](https://www.tensorflow.org/datasets)

## 贡献

如果发现notebook中的问题或有改进建议，请：
1. 检查原始markdown文档
2. 使用转换脚本重新生成
3. 提交问题报告或pull request

## 许可证

请参考原始项目的许可证条款。 