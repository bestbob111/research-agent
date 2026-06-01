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

## 构建向量索引

完整流程：

1. 把 PDF 放到 `/mnt/bigdata/research-agent/papers`
2. 提取 PDF 文本：

```bash
python scripts/ingest_pdfs.py
```

3. 构建索引：

```bash
python scripts/build_index.py --reset --limit 1
```

4. 查询索引：

```bash
python scripts/query_index.py "你的问题"
```

## 本地 RAG 问答

完整流程：

1. 提取 PDF 文本：

```bash
python scripts/ingest_pdfs.py
```

2. 构建向量索引：

```bash
python scripts/build_index.py --reset --max-chars-per-embed 1000
```

3. 查看相似片段：

```bash
python scripts/query_index.py "你的问题"
```

4. 生成基于证据的回答：

```bash
python scripts/ask_local.py "你的问题"
```

`query_index.py` 只显示相似片段；`ask_local.py` 会调用 qwen3:14b 生成基于证据的回答。回答质量取决于 PDF 文本提取质量和索引质量。

故障处理：

- 如果某些 PDF 解析出的 txt 为空，构建索引时会自动跳过。
- 如果某些 chunk embedding 失败，会记录到 `/mnt/bigdata/research-agent/metadata/index_failures.json`。
- 如果某些 chunk 触发 NaN/inf embedding，会被跳过并记录到 `/mnt/bigdata/research-agent/metadata/index_failures.json`。
- 少量失败 chunk 通常可以接受。
- 推荐先运行小批量检查：

```bash
python scripts/build_index.py --reset --limit 1
```

- 如遇到 Ollama 500，可尝试降低单次 embedding 文本长度：

```bash
python scripts/build_index.py --reset --max-chars-per-embed 1000
```

运行测试：

```bash
pytest
```
