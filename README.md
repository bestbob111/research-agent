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

## 统一 CLI 使用方式

安装开发模式：

```bash
pip install -e .
```

常用命令：

```bash
research-agent status
research-agent ingest
research-agent index --reset --max-chars-per-embed 1000
research-agent query "你的问题"
research-agent ask "你的问题"
research-agent review "你的综述问题"
research-agent papers list
research-agent papers list --limit 30
research-agent papers list --offset 30 --limit 30
```

如果只是想看前几条文献，优先使用 `--limit`，而不是管道接 `head`。

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

## 文献元数据管理

`ingest_pdfs.py` 成功生成 txt 后，会自动把基础文献信息写入 `/mnt/bigdata/research-agent/metadata/papers.sqlite`。

查看当前文献列表：

```bash
python scripts/list_papers.py
```

后续 RAG 证据来源会逐步显示标题、作者、年份等文献信息。

## DeepSeek 复杂综述

在 `.env` 中配置：

```bash
DEEPSEEK_API_KEY=你的 API key
```

先构建本地索引，然后运行：

```bash
python scripts/review_with_deepseek.py "请总结这些文献中 SERF 原子磁强计的主要噪声来源和抑制方法"
```

也可以使用统一 CLI：

```bash
research-agent review "请总结 SERF 原子磁强计的主要噪声来源和抑制方法"
```

本地只把检索到的相关证据片段发给 DeepSeek，不会把整个文献库全部上传。review 会自动保存 Markdown 报告，默认位置是 `/mnt/bigdata/research-agent/reports`。报告包含研究问题、生成信息、综合分析、证据来源和使用注意事项。

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
