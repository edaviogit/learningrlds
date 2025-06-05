# SVG 正态分布曲线测试

本文件专门用来测试SVG绘制正态分布曲线的各种方法。

## 测试1: 简单矩形柱状图

<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
  <!-- 坐标轴 -->
  <line x1="50" y1="150" x2="350" y2="150" stroke="black" stroke-width="2"/>
  <line x1="50" y1="150" x2="50" y2="50" stroke="black" stroke-width="2"/>
  
  <!-- 简单柱状图 -->
  <rect x="60" y="140" width="20" height="10" fill="red"/>
  <rect x="90" y="120" width="20" height="30" fill="red"/>
  <rect x="120" y="90" width="20" height="60" fill="red"/>
  <rect x="150" y="70" width="20" height="80" fill="red"/>
  <rect x="180" y="60" width="20" height="90" fill="red"/>
  <rect x="210" y="70" width="20" height="80" fill="red"/>
  <rect x="240" y="90" width="20" height="60" fill="red"/>
  <rect x="270" y="120" width="20" height="30" fill="red"/>
  <rect x="300" y="140" width="20" height="10" fill="red"/>
</svg>

## 测试2: 使用path绘制平滑曲线

<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
  <!-- 坐标轴 -->
  <line x1="50" y1="150" x2="350" y2="150" stroke="black" stroke-width="2"/>
  <line x1="50" y1="150" x2="50" y2="50" stroke="black" stroke-width="2"/>
  
  <!-- 正态分布曲线 -->
  <path d="M 60 140 Q 100 120 140 90 Q 180 60 200 60 Q 220 60 260 90 Q 300 120 340 140" 
        stroke="blue" stroke-width="3" fill="none"/>
</svg>

## 测试3: 使用circle点连线

<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
  <!-- 坐标轴 -->
  <line x1="50" y1="150" x2="350" y2="150" stroke="black" stroke-width="2"/>
  <line x1="50" y1="150" x2="50" y2="50" stroke="black" stroke-width="2"/>
  
  <!-- 数据点 -->
  <circle cx="70" cy="140" r="3" fill="green"/>
  <circle cx="100" cy="120" r="3" fill="green"/>
  <circle cx="130" cy="90" r="3" fill="green"/>
  <circle cx="160" cy="70" r="3" fill="green"/>
  <circle cx="190" cy="60" r="3" fill="green"/>
  <circle cx="220" cy="70" r="3" fill="green"/>
  <circle cx="250" cy="90" r="3" fill="green"/>
  <circle cx="280" cy="120" r="3" fill="green"/>
  <circle cx="310" cy="140" r="3" fill="green"/>
  
  <!-- 连接线 -->
  <polyline points="70,140 100,120 130,90 160,70 190,60 220,70 250,90 280,120 310,140" 
            stroke="green" stroke-width="2" fill="none"/>
</svg>

## 测试4: 最基础的矩形测试

<svg width="200" height="100">
  <rect x="10" y="10" width="50" height="30" fill="blue"/>
  <rect x="70" y="10" width="50" height="30" fill="red"/>
  <rect x="130" y="10" width="50" height="30" fill="green"/>
</svg>

## 测试5: 带填充的正态分布

<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
  <!-- 坐标轴 -->
  <line x1="50" y1="150" x2="350" y2="150" stroke="black" stroke-width="2"/>
  <line x1="50" y1="150" x2="50" y2="50" stroke="black" stroke-width="2"/>
  
  <!-- 填充区域 -->
  <path d="M 60 150 L 60 140 Q 100 120 140 90 Q 180 60 200 60 Q 220 60 260 90 Q 300 120 340 140 L 340 150 Z" 
        fill="lightblue" stroke="blue" stroke-width="2"/>
</svg>

## 测试6: 非常简单的线条测试

<svg width="200" height="100">
  <line x1="0" y1="50" x2="200" y2="50" stroke="black" stroke-width="2"/>
  <line x1="100" y1="0" x2="100" y2="100" stroke="red" stroke-width="2"/>
  <text x="110" y="55" fill="black">Test</text>
</svg>

## 新测试7: 单独测试line元素

<svg width="300" height="150" xmlns="http://www.w3.org/2000/svg">
  <!-- 基础水平线 -->
  <line x1="10" y1="30" x2="100" y2="30" stroke="red"/>
  <!-- 基础垂直线 -->
  <line x1="50" y1="10" x2="50" y2="50" stroke="blue"/>
  <!-- 斜线 -->
  <line x1="120" y1="10" x2="200" y2="50" stroke="green"/>
  <!-- 带宽度的线 -->
  <line x1="10" y1="80" x2="100" y2="80" stroke="purple" stroke-width="3"/>
  <!-- 简单文字 -->
  <text x="10" y="120" fill="black">Basic Lines Test</text>
</svg>

## 新测试8: 测试stroke-dasharray

<svg width="300" height="100" xmlns="http://www.w3.org/2000/svg">
  <!-- 实线 -->
  <line x1="10" y1="20" x2="100" y2="20" stroke="black" stroke-width="2"/>
  <text x="110" y="25" fill="black">Solid</text>
  
  <!-- 虚线 (可能不支持) -->
  <line x1="10" y1="40" x2="100" y2="40" stroke="red" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="110" y="45" fill="red">Dashed</text>
  
  <!-- 点线 (可能不支持) -->
  <line x1="10" y1="60" x2="100" y2="60" stroke="blue" stroke-width="2" stroke-dasharray="2,3"/>
  <text x="110" y="65" fill="blue">Dotted</text>
</svg>

## 新测试9: 测试polyline vs 多个line

<svg width="400" height="150" xmlns="http://www.w3.org/2000/svg">
  <!-- 方法1: polyline -->
  <polyline points="20,30 50,20 80,40 110,25 140,35" 
            stroke="red" stroke-width="2" fill="none"/>
  <text x="20" y="60" fill="red">Polyline</text>
  
  <!-- 方法2: 多个line连接 -->
  <line x1="20" y1="90" x2="50" y2="80" stroke="blue" stroke-width="2"/>
  <line x1="50" y1="80" x2="80" y2="100" stroke="blue" stroke-width="2"/>
  <line x1="80" y1="100" x2="110" y2="85" stroke="blue" stroke-width="2"/>
  <line x1="110" y1="85" x2="140" y2="95" stroke="blue" stroke-width="2"/>
  <text x="20" y="120" fill="blue">Multiple Lines</text>
</svg>

## 新测试10: 不同的path命令

<svg width="400" height="150" xmlns="http://www.w3.org/2000/svg">
  <!-- 简单直线path -->
  <path d="M 20 30 L 80 30" stroke="red" stroke-width="2" fill="none"/>
  <text x="20" y="50" fill="red">Simple Path</text>
  
  <!-- 曲线path -->
  <path d="M 20 80 C 40 60, 60 100, 80 80" stroke="blue" stroke-width="2" fill="none"/>
  <text x="20" y="110" fill="blue">Curve Path</text>
</svg>

## 测试结果检查

请检查以上所有测试，并告诉我：

**基础测试结果**：
- 测试1: 红色矩形柱状图 - ✅/❌
- 测试4: 三个彩色矩形 - ✅/❌  
- 测试6: 基础线条和文字 - ✅/❌

**新增测试结果**：
- 测试7: 各种line元素 - ✅/❌
- 测试8: 虚线效果 - ✅/❌
- 测试9: polyline vs 多个line - ✅/❌  
- 测试10: 不同path命令 - ✅/❌

**问题测试**：
- 测试2: path曲线 - ❌
- 测试3: polyline折线 - ❌  
- 测试5: path填充 - ❌

根据新测试结果，我们可以确定哪些元素完全可用，然后设计最佳的解决方案！ 