# YOLOv12 目标检测应用

基于YOLOv12和PySide6的智能目标检测系统，支持疲劳检测、语音助手、邮件通知等功能。

## 功能特性

- 🎯 YOLOv12目标检测
- 😴 疲劳检测分析
- 🎤 语音助手
- 📧 邮件通知
- 📊 数据可视化
- 💾 检测结果归档
- 🔐 用户登录系统
- 📱 二维码分享

## 系统要求

- Python 3.8+
- Windows/Linux/MacOS
- 支持CUDA的GPU（可选，用于加速推理）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 启动应用

```bash
cd yolov12_pljs_app
python main.py
```

### 配置说明

1. **模型权重文件**
   - 将训练好的模型权重文件放入 `weights/` 目录
   - 支持的格式：`.pt`, `.pth`

2. **配置文件**
   - 复制 `config_template.json` 为 `config.json`
   - 根据实际情况修改配置参数

3. **邮件配置**
   - 在 `credentials.json` 中配置邮件账户信息
   - 支持SMTP协议发送邮件

## 项目结构

```
yolov12_pljs_app/
├── main.py              # 主程序入口
├── main_window.py       # 主窗口界面
├── YOLOv12Model.py      # YOLOv12模型封装
├── ai_analyzer.py       # AI分析器（疲劳检测）
├── voice_assistant.py   # 语音助手
├── mail_sender.py       # 邮件发送
├── login_gui.py         # 登录界面
├── qr_share.py          # 二维码分享
├── IMcore/              # 图像处理核心
│   ├── __init__.py
│   └── IMplots.py
├── ultralytics/         # YOLO工具库
│   ├── __init__.py
│   └── utils/
├── weights/             # 模型权重目录
│   ├── yolov12n.pt
│   ├── yolov12s.pt
│   └── yolov12x.pt
├── archive/             # 检测结果归档
├── data_store.db        # SQLite数据库
└── config.json          # 配置文件
```

## 主要功能说明

### 1. 目标检测
- 支持多种YOLOv12模型（n/s/m/l/x）
- 实时检测和批量检测
- 可视化检测结果

### 2. 疲劳检测
- 基于AI的疲劳状态分析
- 实时监测和预警
- 历史数据记录

### 3. 语音助手
- 语音交互功能
- 指令识别和执行
- 语音反馈

### 4. 邮件通知
- 检测结果邮件发送
- 支持附件和图片
- 定时发送功能

### 5. 数据管理
- SQLite数据库存储
- 检测结果归档
- 数据导出功能

## 注意事项

⚠️ **重要提示：**

1. **模型权重文件**：由于文件较大，模型权重文件未包含在仓库中，请自行下载或训练

2. **敏感信息**：
   - `credentials.json` 包含邮件账户信息，请勿上传到公开仓库
   - `config.json` 可能包含API密钥，请妥善保管

3. **数据库文件**：`data_store.db` 会在首次运行时自动创建

4. **虚拟环境**：建议使用虚拟环境运行项目

## 开发说明

### 代码规范
- 遵循PEP 8代码风格
- 使用类型注解
- 添加必要的注释

### 依赖管理
- 所有依赖列在 `requirements.txt` 中
- 使用虚拟环境隔离依赖

## 常见问题

### 1. 模型加载失败
- 检查模型权重文件路径
- 确认模型文件完整性

### 2. 邮件发送失败
- 检查SMTP配置
- 确认网络连接

### 3. 语音功能异常
- 检查音频设备
- 确认语音库安装

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交Issue或联系开发者。

---

**注意：本项目仅供学习和研究使用。**