# 医学 RAG 爬虫 → Elasticsearch（中文数据源）

一个端到端的医学数据流水线：**抓取 5 个中文医学网站 → 清洗 → 分块 → 向量化 →
写入 Elasticsearch**。每一条记录都会保存原始信息（来源、URL、元数据）、
分块文本及其向量（`dense_vector` + HNSW 余弦检索），可直接用于语义检索（RAG）。

## 数据来源

| key           | 站点                        | 内容                          | 方式                  |
|---------------|-----------------------------|-------------------------------|-----------------------|
| `who_zh`      | WHO 中文实况报道             | 公共卫生事实页（中文）          | HTML 抓取             |
| `msd_cn`      | 默沙东诊疗手册（中文专业版）   | 临床专题文章                   | HTML 抓取 + sitemap   |
| `a_hospital`  | A+医学百科                   | 疾病/医学概念百科词条           | HTML 抓取             |
| `jk39`        | 39健康疾病百科               | 疾病主页 + 病因/症状/预防/就诊子栏目 | HTML 抓取        |
| `xywy`        | 寻医问药疾病库               | 疾病概述（页面为 GBK 编码）      | HTML 抓取             |

所有爬虫都保持克制：遵守 `robots.txt`、带随机抖动的限速；其中默沙东中文站单独
遵循站点的 `Crawl-delay: 5` 指令。

## 项目结构

```
run_pipeline.py            CLI 入口（ingest / search）
medical_rag/
  config.py                全局配置（Elasticsearch、embedding、分块、抓取）
  es_store.py              ES 客户端 + 索引 mapping + 批量写入 + kNN 检索
  embedder.py              GLM API（推荐）/ sentence-transformers / TF-IDF+LSA 回退
  cleaners.py              HTML→文本、空白/CJK 规范化、语言识别
  chunking.py              中英文句界感知的重叠分块器
  pipeline.py              抓取 → 清洗 → 分块 → 向量化 → 入库；+ search()
  crawling/
    base.py                BaseCrawler、RawDoc、HTTP/robots 工具
    who_zh.py  msd_cn.py  a_hospital.py  jk39.py  xywy.py
    registry.py            站点 key → 爬虫类的映射
```

**索引结构（Elasticsearch）** —— 两个索引（mapping 定义在 `es_store.py`），
原始来源信息、分块文本与向量都保留：

- `medical_documents` —— `source`、`source_name`、`source_url`、`title`、
  `page_number`、`lang`、`doc_metadata`（object）、`raw_text`、`created_at`。
- `medical_chunks` —— `document_id`、`chunk_index`、`chunk_text`、`char_length`、
  `token_count`、`embedding`（`dense_vector(N)`，`similarity: cosine`，
  `index: true` 自动维护 HNSW 图）、`created_at`；`source` / `source_url` /
  `title` / `lang` 冗余存储在每个分块里，检索无需 join。

  文档与分块的 `_id` 都是确定性哈希（`sha256(source|source_url)` /
  `sha256(document_id#chunk_index)`），重复录入等于幂等 upsert。

## 环境搭建

1. **Elasticsearch（Docker，开启安全认证 + 数据卷持久化）。** 官方镜像默认启用
   TLS 与密码认证。启动一个单节点集群，`data` 与 `config` 分别挂载 named
   volume——索引数据、TLS 证书与 keystore（密码等 secure settings）都落在卷里，
   删除容器后用同一条命令重建即可原样恢复，密码与证书不会变：

   ```bash
   docker run -d --name med-es -p 9200:9200 \
     -e discovery.type=single-node \
     -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
     --ulimit nofile=65536:65536 \
     -v med-es-data:/usr/share/elasticsearch/data \
     -v med-es-config:/usr/share/elasticsearch/config \
     docker.elastic.co/elasticsearch/elasticsearch:9.5.4
   ```

   > `config` 卷很关键：keystore 里的加密密钥与 SSL 配置和 `data` 里的集群
   > 状态是配套的，只挂 `data` 不挂 `config`，重建容器后会因密钥不匹配进入
   > degraded 状态且 TLS 不生效。

   首次启动后重置 `elastic` 超级用户密码，并导出 HTTP 层的自签 CA 证书
   （仅在**第一次**或彻底重置后需要；此后重建容器不必重复）：

   ```bash
   # 等待集群就绪后执行；命令会打印新密码
   docker exec med-es elasticsearch-reset-password -u elastic -b -s

   # 导出 CA 证书，客户端用 https 连接时校验
   mkdir -p certs
   docker cp med-es:/usr/share/elasticsearch/config/certs/http_ca.crt certs/
   ```

   验证：`curl --cacert certs/http_ca.crt -u elastic:<密码> https://localhost:9200/_cluster/health`

   彻底清空数据（连密码证书一起重置）：`docker rm -f med-es && docker volume rm med-es-data med-es-config` 后重新执行上面的 `docker run`。

1b. **（可选）Kibana 可视化。** 通过 service account token 连接 ES
   （Kibana 8+ 不允许用 `elastic` 超级用户作为服务连接账号），UI 登录仍用
   `elastic` + 密码：

   ```bash
   # 一次性准备：容器网络 + service token（token 写入 .env 的 KIBANA_SERVICE_TOKEN）
   docker network create esnet
   docker network connect esnet med-es
   docker exec med-es elasticsearch-service-tokens create elastic/kibana kibana-token
   #   输出形如 "SERVICE_TOKEN elastic/kibana/kibana-token = AAEAAWVsYXN0aWMva2li..."，
   #   把 "=" 后面的整串（不含空格）存进 .env：KIBANA_SERVICE_TOKEN=<token>

   docker run -d --name med-kibana --network esnet -p 5601:5601 \
     --memory=1600m -e "NODE_OPTIONS=--max-old-space-size=1024" \
     -e ELASTICSEARCH_HOSTS=https://med-es:9200 \
     -e ELASTICSEARCH_SERVICEACCOUNTTOKEN=$KIBANA_SERVICE_TOKEN \
     -e ELASTICSEARCH_SSL_CERTIFICATEAUTHORITIES=/usr/share/kibana/config/certs/http_ca.crt \
     -e ELASTICSEARCH_SSL_VERIFICATIONMODE=certificate \
     -v med-kibana-data:/usr/share/kibana/data \
     -v "$PWD/certs/http_ca.crt:/usr/share/kibana/config/certs/http_ca.crt:ro" \
     docker.elastic.co/kibana/kibana:9.5.4
   ```

   启动约 1–3 分钟后访问 `http://localhost:5601`，用 `elastic` + ES 密码登录。
   **Discover** 里为 `medical_chunks` / `medical_documents` 创建 data view 即可
   浏览全部数据（按 `source` 过滤、查看 `chunk_text` 全文）；**Dev Tools** 可
   交互式执行 DSL / kNN 查询。512MB 的 Node 堆会导致启动时 OOM，务必保持
   `--max-old-space-size` ≥ 1024。

2. **Python 环境。**

   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt           # 包含 sentence-transformers（默认）
   # 若无法安装重型 ML 依赖，管道会自动回退到 TF-IDF + LSA 向量化，
   # 仍可生成可用向量并支持检索。
   ```

   > **模型下载（HuggingFace）。** 如果你的网络无法访问 huggingface.co（国内环境
   > 常见），需要在运行前设置 `HF_ENDPOINT` 镜像；代码在未设置时会默认使用该值：
   >
   > ```bash
   > export HF_ENDPOINT=https://hf-mirror.com
   > ```

3. **配置项（环境变量，以下为默认值）：**

   | 变量 | 默认值 | 含义 |
   |------|--------|------|
   | `ES_URL` | `http://localhost:9200` | ES 地址（开启安全认证时为 `https://localhost:9200`） |
   | `ES_USER` / `ES_PASSWORD` | （空） | ES 账号 / 密码（basic auth） |
   | `ES_CA_CERT` | （空） | CA 证书路径（相对项目根，如 `certs/http_ca.crt`） |
   | `ES_DOCS_INDEX` / `ES_CHUNKS_INDEX` | `medical_documents` / `medical_chunks` | 索引名 |
   | `EMBED_BACKEND` | `auto` | `glm` / `sentence-transformers` / `tfidf` / `auto` |
   | `EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | HuggingFace 模型 id |
   | `EMBED_DIM` | `1024` | 向量维度（GLM 与 TF-IDF 路径使用；GLM embedding-3 支持 256–2048） |
   | `GLM_API_KEY` | （无） | 智谱 GLM embedding API 密钥（**必填才能用 glm 后端**） |
   | `GLM_EMBED_MODEL` | `embedding-3` | GLM embedding 模型名 |
   | `GLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4/embeddings` | GLM API 端点 |
   | `GLM_EMBED_BATCH` | `32` | 每次 API 请求携带的文本条数 |
   | `CHUNK_SIZE` | `512` | 目标分块 token 数（LlamaIndex SentenceSplitter） |
   | `CHUNK_OVERLAP` | `100` | 相邻分块重叠 token 数 |
   | `CRAWL_MAX_PAGES` | `15` | 每个站点最多抓取文档数 |
   | `CRAWL_DELAY` | `1.0` | 两次请求之间的间隔（秒） |

   > 默认模型 `BAAI/bge-small-zh-v1.5`（512 维）对中文友好。若你的语料以英文为主，
   > 可将 `EMBED_MODEL` 设为 `sentence-transformers/all-MiniLM-L6-v2`（384 维），或
   > 使用多语种模型如 `paraphrase-multilingual-MiniLM-L12-v2`。

4. **GLM embedding（推荐）。** `glm` 后端调用智谱 AI 的 embedding API
   （默认 `embedding-3`，512 维），无需本地 ML 栈。API key 属于敏感信息，
   **不要写进代码提交到仓库**——放在环境变量或项目根目录的 `.env` 文件中
   （`.env` 已被 `.gitignore` 忽略，程序启动时会自动加载）：

   ```bash
   cat > .env <<'EOF'
   GLM_API_KEY=你的密钥

   ES_URL=https://localhost:9200
   ES_USER=elastic
   ES_PASSWORD=重置得到的密码
   ES_CA_CERT=certs/http_ca.crt
   EOF
   # 或者直接导出环境变量：
   export GLM_API_KEY=你的密钥
   ```

   `EMBED_BACKEND=auto`（默认）在检测到 `GLM_API_KEY` 时会优先使用 GLM，
   否则依次回退到 sentence-transformers、TF-IDF。

## 使用方法

```bash
# 抓取全部 5 个中文站点并入库（有 GLM_API_KEY 时默认走 GLM embedding）。
python run_pipeline.py ingest

# 只抓取 WHO 中文，减少页面数：
python run_pipeline.py ingest --sites who_zh --max-pages 8

# 强制使用轻量的 TF-IDF 向量化：
python run_pipeline.py ingest --embed-backend tfidf

# 入库前先删除并重建表结构（干净重建）：
python run_pipeline.py ingest --recreate

# 对库内分块做语义检索：
python run_pipeline.py search --query "2型糖尿病的治疗" --k 5
英文查询同样支持，例如：python run_pipeline.py search --query "treatment for type 2 diabetes" --k 5
```

`search` 命令会返回最匹配的分块，包含分块文本、来源 URL、标题以及余弦
**相似度** 得分。

## 各部分如何工作

- **清洗**（`cleaners.py`）：剔除 script/style/nav 等样板内容，优先取
  `<article>` / `<main>` / `#content` 区域，压缩空白、还原 HTML 实体、去除重复行、
  过滤垃圾字节；用一个轻量的 CJK 比例启发式判定 `zh` / `en`。
- **分块**（`chunking.py`）：基于 **LlamaIndex `SentenceSplitter`**——中文句界感知
  （`。！？；`）、按 token 预算贪心填充、相邻块携带重叠。自定义中文 tokenizer
  （中文 1 字 = 1 token、英文 1 词 = 1 token）避免了 tiktoken 运行时下载，
  国内网络离线可用。`CHUNK_SIZE=512` / `CHUNK_OVERLAP=100` 单位为该 token 计数，
  过短块（< 80 字符）会被丢弃并重新编号。
- **向量化**（`embedder.py`）：默认优先调用 GLM embedding API（远程语义向量，
  需 `GLM_API_KEY`）；无 key 时回退到 `sentence-transformers`（本地语义向量），
  ML 依赖仍不可用时最终回退到 `TF-IDF + LSA`（scikit-learn）。所有后端都会做 L2
  归一化并补齐到配置的宽度，保证 `vector(N)` 列一致。当前生效的后端、模型与维度会
  被持久化到 `data/processed/`，使 `search` 复用同一向量空间。
- **入库**（`es_store.py` + `pipeline.py`）：每个页面/文章对应一条
  `medical_documents`（含来源信息），对应多条 `medical_chunks`（各自携带
  `dense_vector` 向量，mapping 自动维护 HNSW 图）。整个运行**幂等** ——
  已入库的 `(source, source_url)` 在下次运行时会被跳过。检索走 ES 的 kNN
  查询（余弦相似度，ES 得分 `(1+cos)/2` 已换算回余弦值）。

## 备注

- 5 个站点均为 HTML 抓取且遵守 `robots.txt`，全部使用礼貌的抓取间隔；默沙东中文站
  遵循其 `Crawl-delay: 5`。候选站 wiki8.com（医学百科）因域名失效被排除。
- 各爬虫会把额外的来源信息（实况报道 slug、疾病拼音码/ID、所属科室分类等）写入
  `documents.doc_metadata`；jk39 会把疾病四个子栏目页合并为一篇文档。
- 寻医问药（xywy）页面为 **GBK 编码**，爬虫内部已显式按 GBK 解码。
- 抓取到的原始页面缓存于 `data/raw/`，默沙东中文站的 sitemap 缓存于
  `data/processed/`，以避免每次运行都重复下载约 10 MB 的数据。
