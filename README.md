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
research-agent import-downloads --source arxiv
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

## 研究主题管理

```bash
research-agent topic create "SERF 原子磁强计噪声抑制" --description "调研 SERF 磁强计的噪声来源、抑制方法和应用"
research-agent topic list
research-agent topic show 1
research-agent topic plan 1
research-agent topic plan 1 --force
research-agent topic plan 1 --deepseek --n-evidence 15
research-agent topic report 1
research-agent topic report 1 --local
research-agent topic report 1 --n-evidence 30
```

topic 功能用于长期管理研究方向，后续可与检索计划、报告和文献筛选关联。`topic create` 只创建研究主题工作区；`topic plan` 会结合本地文献库检索结果生成研究计划，默认使用本地 qwen3:14b，`--deepseek` 可用于生成更高质量计划。计划文件保存在 `metadata/topics/topic_<id>_plan.md`。后续网页检索、知网检索会基于这个计划展开。

`topic report` 会基于 topic plan 和本地文献库证据生成正式调研报告，默认使用 DeepSeek，`--local` 可完全本地生成。报告保存在 `reports_dir`，topic 首页会记录最新报告路径。

## 问题探索模式

```bash
research-agent explore "SERF 原子磁强计的主要噪声来源有哪些，分别如何抑制？"
research-agent explore "SERF 原子磁强计的主要噪声来源有哪些，分别如何抑制？" --local
research-agent explore "SERF 原子磁强计的主要噪声来源有哪些，分别如何抑制？" --topic-id 1 --rounds 2 --n-evidence 10
research-agent explore "SERF 原子磁强计的主要噪声来源有哪些，分别如何抑制？" --assess-with-deepseek
research-agent explore list
research-agent explore show explore_YYYYmmdd_HHMMSS
research-agent explore continue explore_YYYYmmdd_HHMMSS --local
```

`explore` 会主动拆解问题，多次检索本地文献库，并保存 Markdown 报告和 JSON 结构化记录。当前版本只探索本地文献库，不会自动访问知网或网页；后续网页检索会基于 explore 生成的 gaps/search_queries 扩展。

`explore list` 查看历史探索，`explore show` 查看某次探索摘要，`explore continue` 会基于上一次的 gaps/search_queries 继续探索。

explore 报告会包含证据质量评估，`recommended_reading` 可作为精读清单。默认使用本地模型做证据评估，`--assess-with-deepseek` 可用 DeepSeek 做更严格评估。

## arXiv 开放文献检索

```bash
research-agent arxiv search "SERF atomic magnetometer noise" --max-results 20
research-agent arxiv search "SERF atomic magnetometer noise" --max-results 20 --download --download-limit 3
```

arXiv search 只使用公开 API。检索结果会保存到 `metadata_dir`，并导入 `papers.sqlite`。下载的 PDF 放到 `downloads_dir/arxiv`。下载后如需纳入本地 RAG，需要把 PDF 导入 `papers_dir` 后运行 ingest/index。

## 导入下载 PDF

```bash
research-agent arxiv search "SERF atomic magnetometer noise" --max-results 5 --download --download-limit 2
research-agent import-downloads --source arxiv
research-agent index --reset --max-chars-per-embed 1000
```

arXiv 下载的 PDF 默认保存在 `downloads_dir/arxiv`。用户手动下载的 PDF 可以放到 `downloads_dir/import`，再运行：

```bash
research-agent import-downloads --source import
```

`import-downloads` 会把 PDF 复制到 `papers_dir`，提取文本到 `texts_dir`，并写入 `metadata_dir/papers.sqlite`。导入后需要重新运行 `research-agent index`，文献才会进入本地 RAG。后续会增加增量索引，避免每次全量重建。

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
