# SVG 支持测试

## 🧪 简单SVG测试

如果您能看到下面的蓝色圆圈，说明您的Markdown预览器支持内嵌SVG：

<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" stroke="blue" stroke-width="3" fill="lightblue"/>
  <text x="50" y="55" text-anchor="middle" fill="darkblue">SVG OK</text>
</svg>

## 📊 简单的柱状图测试

<svg width="300" height="200" style="border: 1px solid #ccc;">
  <rect x="50" y="150" width="40" height="30" fill="#3498db"/>
  <rect x="100" y="100" width="40" height="80" fill="#3498db"/>
  <rect x="150" y="50" width="40" height="130" fill="#3498db"/>
  <rect x="200" y="100" width="40" height="80" fill="#3498db"/>
  <rect x="250" y="150" width="40" height="30" fill="#3498db"/>
  
  <text x="70" y="195" text-anchor="middle" font-size="12">A</text>
  <text x="120" y="195" text-anchor="middle" font-size="12">B</text>
  <text x="170" y="195" text-anchor="middle" font-size="12">C</text>
  <text x="220" y="195" text-anchor="middle" font-size="12">D</text>
  <text x="270" y="195" text-anchor="middle" font-size="12">E</text>
</svg>

## 🎨 CSS替代方案

### 正态分布 (CSS版)
<div style="display: flex; align-items: end; height: 150px; background: #f8f9fa; padding: 20px; border: 1px solid #ddd;">
  <div style="width: 20px; height: 20px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 40px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 70px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 100px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 130px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 150px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 130px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 100px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 70px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 40px; background: #3498db; margin: 1px;"></div>
  <div style="width: 20px; height: 20px; background: #3498db; margin: 1px;"></div>
</div>

### 右偏分布 (CSS版)
<div style="display: flex; align-items: end; height: 150px; background: #fff5f5; padding: 20px; border: 1px solid #ddd;">
  <div style="width: 20px; height: 30px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 80px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 150px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 120px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 90px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 70px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 55px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 40px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 30px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 20px; background: #e67e22; margin: 1px;"></div>
  <div style="width: 20px; height: 15px; background: #e67e22; margin: 1px;"></div>
</div>

## 📋 如果都看不到图形

那就说明您的Markdown预览器不支持HTML样式，这种情况下ASCII图表是唯一选择：

```
正态分布:
   ●
 ● ● ●
● ● ● ● ●
───────────
15 25 35 45 55

右偏分布:
●
● ●
● ● ●
● ● ● ●
─────────────
30 60 90 120 150
```

## 🔧 推荐解决方案

1. **安装更好的扩展**: `Markdown Preview Enhanced`
2. **使用外部工具**: 在线Markdown编辑器如 Typora、Mark Text
3. **浏览器预览**: 将MD文件在浏览器中打开
4. **接受ASCII版本**: 使用纯文本图表

请告诉我您在这个测试文件中能看到哪些图形！ 