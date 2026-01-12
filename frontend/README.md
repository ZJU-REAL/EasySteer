# EasySteer - Steer Vector 控制面板

EasySteer 是一个美观易用的前端界面，用于配置和管理 VLLM 的 Steer Vector 功能，并直接生成文本结果。

## 功能特点

- 🎨 **现代化界面**：采用渐变色和卡片式设计，界面美观大方
- 🚀 **交互友好**：实时参数验证，滑块控制，文件选择器
- ⚡ **快捷操作**：支持键盘快捷键（Ctrl+Enter 生成，Ctrl+R 重置）
- 📱 **响应式设计**：完美适配桌面和移动设备
- 🔧 **完整功能**：支持所有 Steer Vector 参数配置
- 🌐 **多语言支持**：支持中文和英文界面，可一键切换（默认英文）
- 🤖 **实时生成**：直接在界面中生成并显示结果

## 主要功能

### 模型配置
- **模型路径**：指定要使用的模型路径
- **GPU 设备**：选择要使用的 GPU（支持多 GPU）
- **标准化选项**：是否标准化 Steer Vector
- **输入指令**：直接输入要生成的提示词

### 采样参数
- **Temperature**：控制生成的随机性（0-2）
- **Max Tokens**：最大生成 token 数
- **Repetition Penalty**：重复惩罚系数

### Steer Vector 配置
- **算法选择**：Direct 或 LoReft (Low-rank Linear Subspace Representation Finetuning)
- **层级控制**：精确指定目标层级
- **触发器配置**：支持预填充和生成阶段的触发条件
- **调试模式**：实时查看调试信息

## 快速开始

> **注意**: EasySteer 仅支持在 Linux 和 macOS 系统上运行，不支持 Windows。

### 1. 安装依赖

```bash
cd frontend
pip install -r requirements.txt
```

确保已安装 VLLM：
```bash
pip install vllm
```

### 2. 使用启动脚本（推荐）

最简单的方式是使用提供的启动脚本：

```bash
chmod +x start.sh
./start.sh
```

启动脚本会自动：
- 检查并安装依赖
- 启动后端服务器（默认端口 5000）
- 启动前端静态服务器（默认端口 8111）
- 自动打开浏览器

### 3. 手动启动（可选）

如果需要手动启动，可以分别运行：

```bash
# 终端1: 启动后端服务器
python app.py

# 终端2: 启动前端服务器
python -m http.server 8111
```

然后在浏览器中访问 `http://localhost:8111/index.html`

## 使用说明

### 语言切换

- 界面右上角提供语言切换按钮
- 支持中文（简体）和英文，默认显示英文
- 语言偏好会自动保存，下次访问时自动应用
- 后端错误消息也会根据选择的语言返回相应的翻译

### 基本使用流程

1. **配置模型**
   - 输入模型路径（如 `/path/to/Qwen2.5-1.5B-Instruct/`）
   - 选择 GPU 设备（如 `0` 或 `0,1,2`）
   - 选择是否标准化 Steer Vector
   - 输入您的提示词或问题

2. **设置采样参数**
   - 调整 Temperature（0 表示确定性生成）
   - 设置最大生成 token 数
   - 调整重复惩罚系数

3. **配置 Steer Vector**
   - 输入 Steer Vector 名称和 ID
   - 指定 Steer Vector 文件路径
   - 选择算法（Direct 或 LoReft）
   - 设置缩放因子（支持负值）
   - 配置目标层级和触发器

4. **生成文本**
   - 点击"Generate"按钮
   - 等待生成完成
   - 在下方查看生成结果

### API 端点

后端服务器提供以下 REST API 端点：

- `POST /api/generate` - 使用 Steer Vector 生成文本
- `POST /api/steer-vector` - 创建新的 steer vector 配置
- `GET /api/steer-vector/<id>` - 获取特定的 steer vector 配置
- `GET /api/steer-vectors` - 列出所有活跃的 steer vector
- `DELETE /api/steer-vector/<id>` - 删除 steer vector 配置
- `GET /api/health` - 健康检查

**注意**：API 支持通过 `Accept-Language` 头部返回相应语言的错误消息

## 预定义配置

前端界面提供了预定义的配置选项，可以快速导入不同场景的 Steer Vector 设置：

- **Emoji LoReft 配置** - 用于表情符号生成的 LoReft 算法配置
- **Emotion Direct 配置** - 用于情感控制的 Direct 算法配置

点击"Import"按钮可以快速加载预定义配置，然后根据需要进行调整。

## 系统要求

- **操作系统**: Linux 或 macOS
- **Python**: 3.8+
- **CUDA**: 推荐 CUDA 11.8+（用于 GPU 加速）
- **注意**: 本项目不支持在 Windows 上部署

## 文件结构

```
frontend/
├── index.html      # 主界面
├── style.css       # 样式文件
├── script.js       # 前端逻辑
├── i18n.js         # 多语言支持
├── app.py          # Flask 后端服务器（包含 VLLM 集成）
├── requirements.txt # Python 依赖
├── start.sh        # Linux/macOS 启动脚本
├── config.py       # 后端配置文件
├── core/           # 核心模块
└── README.md       # 本文档
```

## 自动设置的参数

以下参数会自动设置，无需用户配置：
- `enable_steer_vector`: 始终为 `True`
- `enforce_eager`: 始终为 `True`
- `steer_vector_intervention_level`: 始终为 `"decoder_layer"`
- `tensor_parallel_size`: 根据指定的 GPU 数量自动设置

## 开发说明

### 自定义样式

可以通过修改 `style.css` 中的 CSS 变量来自定义颜色主题：

```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #48bb78;
    --error-color: #f56565;
}
```

### 添加新语言

要添加新的语言支持：

1. 在 `i18n.js` 的 `translations` 对象中添加新语言
2. 在 `index.html` 中添加新的语言切换按钮
3. 在 `app.py` 的 `error_messages` 中添加对应的错误消息翻译

### 扩展功能

要添加新的参数或功能：

1. 在 `index.html` 中添加相应的表单元素
2. 在 `i18n.js` 中添加相关的翻译键值
3. 在 `script.js` 中更新 `submitConfiguration()` 函数
4. 在 `app.py` 中更新后端处理逻辑

## 注意事项

- 确保模型路径正确且模型文件存在
- 确保 steer vector 文件路径正确且文件存在
- ID 必须是唯一的数字
- 使用 `-1` 作为 token ID 表示应用到所有 token
- 支持负数索引来指定位置（如 -1 表示最后一个位置）
- 缩放因子支持负值（如 -2.0）

## 故障排除

1. **无法连接到服务器**：确保后端服务器正在运行（`python app.py`）
2. **模型加载失败**：检查模型路径是否正确，确保有足够的 GPU 内存
3. **文件不存在错误**：检查 Steer Vector 文件路径是否正确
4. **CORS 错误**：确保使用 HTTP 服务器而非直接打开 HTML 文件
5. **语言切换不生效**：清除浏览器缓存或检查 localStorage 是否被禁用
6. **GPU 错误**：确保指定的 GPU 设备存在且可用

## 性能提示

- 模型会在首次使用时加载并缓存，后续使用相同模型会更快
- 使用多 GPU 时，确保 `tensor_parallel_size` 与 GPU 数量匹配
- Temperature 设为 0 可获得确定性输出

## 许可证

本项目遵循 Apache 2.0 许可证 