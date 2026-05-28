# research-agent

本项目是一个本地科研文献助手，目标是围绕本地文献库构建可控、可扩展的科研工作流。

后续计划支持：

- PDF 文献解析
- 本地 embedding 建索引
- Ollama + qwen3:14b 本地问答
- DeepSeek API 复杂综述
- Zotero、知网检索、确认式下载和自动报告扩展

## 当前阶段

当前仓库处于项目初始化阶段，已包含基础目录结构、默认配置、配置加载逻辑、路径检查脚本和基础测试。

## 路径约定

代码位于当前项目目录。

所有大文件和运行数据默认放在 4T 数据盘：

```text
/mnt/bigdata/research-agent
```

默认数据目录：

- PDF 文献：`/mnt/bigdata/research-agent/papers`
- 解析文本：`/mnt/bigdata/research-agent/texts`
- 向量库：`/mnt/bigdata/research-agent/chroma`
- 元数据：`/mnt/bigdata/research-agent/metadata`
- 报告：`/mnt/bigdata/research-agent/reports`
- 下载文件：`/mnt/bigdata/research-agent/downloads`
- Zotero 数据：`/mnt/bigdata/research-agent/zotero`
- 日志：`/mnt/bigdata/research-agent/logs`

不要把 PDF、向量库、报告或下载文件默认放在项目目录中。

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 基本使用

复制环境变量模板：

```bash
cp .env.example .env
```

检查数据目录：

```bash
python scripts/check_paths.py
```

## PDF 文本提取

把 PDF 文件放到：

```text
/mnt/bigdata/research-agent/papers
```

运行：

```bash
python scripts/ingest_pdfs.py
```

提取后的文本会输出到：

```text
/mnt/bigdata/research-agent/texts
```

## Ollama 本地模型检查

需要先安装 Ollama，并拉取默认模型：

```bash
ollama pull qwen3:14b
ollama pull bge-m3
```

运行检查：

```bash
python scripts/check_ollama.py
```

运行测试：

```bash
pytest
```
