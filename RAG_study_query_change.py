# RAG 系统实现（基于阿里云百炼 + Chroma）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 【离线阶段】（只做一次）
#   文档 → 加载 → 分块 → 向量化 → 存入向量数据库

# 【在线阶段】（每次提问）
#   用户提问 → 向量化 → 在向量数据库中检索相似片段 → 塞进 prompt → 模型回答
from langchain_community.document_loaders import TextLoader # 文本加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter #智能分块器
from chromadb import EphemeralClient # 内存版ChromaDB(重启丢失)
from langchain_core.prompts import ChatPromptTemplate #提示词模板
from langchain_openai import ChatOpenAI # LangChain OpenAI 接口
from langchain_core.output_parsers import StrOutputParser # 输出解析器
from collections import Counter # 词频计数器

import httpx # 异步 HTTP 客户
import math # 数学函数（用于 BM25 公式）
import re # 正则表达式（用于中文分词）

LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = 'sk-882d93e0832348979b1a3f1702bac021'
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

# ============ 简易 BM25 关键词检索 ============
class SimpleBM25:
    def __init__(self, documents: list[str]):
        self.documents = documents
        # 预处理：分词（中文按字切，英文按词切）
        self.doc_tokens = [self._tokenize(doc) for doc in documents] # 分词结果 结构: [['机','器','学','习'], ['深','度','学','习'], ...]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens] # 文档词频 [Counter({'机':1, '器':1, ...}), Counter({...}), ...]

        self.avg_dl = sum(len(t) for t in self.doc_tokens) / len(documents) # 平均分档长度
        self.doc_count = len(documents) # 文档总数 (N)
        # 倒排！！！ 在 RAG 中，倒排索引是精准关键词检索的唯一高效方案，与向量检索互补构成工业级系统。你的代码中 inverted_index 和 doc_freqs 正是实现 BM25 的核心，缺一不可。
        self.inverted_index = {} # {词: [文档ID列表]}
        for (i, tokens) in enumerate(self.doc_tokens):
            for token in set(tokens):# 避免重复计数
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(i)
#    → 例: {'学': [0,1,2], '习': [0,1,2], '机': [0], ...}

    @staticmethod
    def _tokenize(text: str)->list[str]:
        """简易分词：中文按字，英文按词"""
        # 英文单词
        words = re.findall(r'[a-zA-Z]+', text)
        # 中文字符（逐字）
        chars = re.findall(r'[\u4e00-\u9fff]', text)
        return words + chars
    #  混合分词结果 (例: ['RAG', '系', '统'])
    
    def search(self, query: str, top_k: int = 5)->list[int]:
        """返回相关文档的索引列表，按 BM25 分数降序"""
        query_tokens = self._tokenize(query) # query 分词
        scores = [0.0] * self.doc_count # 初始化得分数组 [0,0,0,...]
        for qt in query_tokens:
            if qt not in self.inverted_index:
                continue
            doc_ids = self.inverted_index[qt] #词 qt 对应的文档列表
            df = len(doc_ids) # 文档频率 (df)
            # IDF
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            for doc_id in doc_ids:
                tf = self.doc_freqs[doc_id][qt] # 词频 (tf)
                dl = len(self.doc_tokens[doc_id])  # 当前文档长度
                # BM25 公式 对长文档自动降权（dl / avg_dl 部分），避免长文档占据高位
                score = idf * (tf * 1.5 + 0.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / self.avg_dl))
                scores[doc_id] += score
        # 返回 top_k 个文档索引 按得分排序 → 返回文档ID列表 文档索引列表（非内容），需配合 all_docs 获取实际内容

        ranked = sorted(range(self.doc_count), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]
    

# ============ 加载文档 + 分块============ 
loader = TextLoader('knowledge_50.txt', encoding='utf-8')
documents = loader.load()
print(f"加载了 {len(documents)} 个文档")
print(f"总字符数: {len(documents[0].page_content)}")

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " "],# 按中文语义分块 优先级：段落>句子>逗号）
    chunk_size=200,       # 每块最多 200 字符
    chunk_overlap=20 # 块间重叠20字符防断句
)

chunks = splitter.split_documents(documents)
print(f'\n分成了{len(chunks)}个块')

for i,chunk in enumerate(chunks):
    print(f'\n--—块{i}————')
    print(chunk.page_content[:80]+'...')
# ============ 3. 向量化（直接调阿里云 API）============
def get_embeddings(texts: list[str]) -> list[list[float]]:
    """调用阿里云百炼 Embedding API（自动分批，每批最多 10 条）"""
    BATCH_SIZE = 10
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        resp = httpx.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "text-embedding-v3",# text-embedding-v3 返回 1536维向量（适用于中文优化）
                "input": batch,  # 必须是字符串列表
            },
            timeout=30,
        )
        # ↓ 加这行，看具体报错
        if resp.status_code != 200:
            print(f"  ❌ 第 {i//BATCH_SIZE + 1} 批报错: {resp.text}")
        resp.raise_for_status()
        data = resp.json()
        # 按 index 排序，保证顺序一致 百炼API返回结果顺序可能乱 → 必须按index排序保证与输入对应
        data["data"].sort(key=lambda x: x["index"])
        all_vectors.extend([item['embedding'] for item in data["data"]])# 返回值：2D列表，len(vectors)=len(texts), len(vectors[0])=1536
    return all_vectors

# 直接调用阿里云百炼API（绕过LangChain封装，更透明）
# 支持批量向量化（texts 为字符串列表）
# 自动排序保证结果与输入顺序一致

print("正在向量化...")
vectors = get_embeddings([c.page_content for c in chunks])
print(f"✅ 向量化完成，每条 {len(vectors[0])} 维")
# ============ 4. 存入向量数据库 ============

client = EphemeralClient() # 内存数据库（重启丢失）
collection = client.get_or_create_collection(name="company_kb")

ids = [f"chunk_{i}" for i in range(len(chunks))]
documents_text = [c.page_content for c in chunks] # 原文
metadatas = [{"source": "knowledge.txt", "index": i} for i in range(len(chunks))]

collection.add(
    ids=ids,
    documents=documents_text,
    metadatas=metadatas,
    embeddings=vectors,  # 直接传入向量，Chroma 不会再调默认 embedding
)

print(f"✅ 已存入向量数据库，共 {collection.count()} 条")

# ============ 重排函数 ============
"""
BM25/向量检索仅保证召回相关性，LLM重排可解决：
✅ 语义歧义（如“苹果”指水果还是公司）
✅ 关键信息位置（段落开头的词更重要）
"""
def rerank_with_llm(query: str, documents: list[str], top_k: int = 2) -> list[str]:
    """
    用 LLM 对候选文档做相关性排序（不需要额外 Rerank 模型权限）
    """
    # 把文档编号，让 LLM 只返回排序
    numbered_docs = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(documents))
    
    prompt = f"""以下是用户问题和若干候选文档（已编号）。请判断每个文档与问题的相关性，按相关性从高到低排序。
    只输出编号列表，如：0, 2, 1, 3, 4

    用户问题：{query}

    候选文档：
    {numbered_docs}

    相关性排序（从高到低）：
    """

    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-turbo",  # 用 turbo 更快更便宜
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,       # 确定性输出
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    result_text = data["choices"][0]["message"]["content"].strip()

    # 解析 "0, 2, 1" 这样的输出
    indices = [int(x.strip()) for x in result_text.split(",") if x.strip().isdigit()]
    
    print(f"  [Rerank] 候选 {len(documents)} 条 → 排序: {indices} → 取 top-{top_k}")
    for i in indices[:top_k]:
        print(f"    #{i} | {documents[i][:40]}...")

    return [documents[i] for i in indices[:top_k]]
# ============ RRF 融合 ============
"""
RRF 工作原理：
公式：score = 1/(k + rank + 1)
k=60 是平滑参数（增大k使结果更平均）
优势：无需归一化不同检索器的分数，直接融合排序位置
"""
def rrf_fusion(rank_lists: list[list[int]], k:int=60,top_k:int=3):

    """
        Reciprocal Rank Fusion 融合多路检索结果
        
        参数：
            rank_lists: 多路检索结果的文档索引列表，如 [[0,3,1,4,2], [2,0,4,1,3]]
            k: 平滑参数（通常 60）
            top_k: 最终返回几条
        
        返回：
            融合后的文档索引列表
        """
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)

    # 按分数降序
    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked[:top_k]
# ============ 混合检索函数 ============
def rewrite_query(query: str) -> str:
    """用 LLM 将口语化问题改写为检索友好格式"""
    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-turbo",
            "messages": [{"role": "user", "content": f"将问题改写为更正式、专业的格式，只输出问题：{query}"}],
            "temperature": 0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    rewritten_query = data["choices"][0]["message"]["content"].strip()
    print(f"  [问题改写] {query} → {rewritten_query}")
    return rewritten_query


def expand_query(query: str, n: int = 3) -> list[str]:# 问题扩展（1→N）
    """将一个复杂问题拆成 N 个子问题，返回改写后的问题列表"""
    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-turbo",
            "messages": [
                {"role": "system", "content": (
                    f'将用户问题拆成 {n} 个子问题，每行一个，只输出问题，不要编号不要解释。'
                )},
                {"role": "user", "content": query}],
            "temperature": 0,
        },
        timeout=15,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    sub_queries = [line.strip() for line in text.split('\n') if line.strip()]
    print(f"  [问题改写] {query} → {sub_queries}")
    return sub_queries[:n]

def hyde(query: str) -> str:
    """
        HyDE: Hypothetical Document Embeddings
        让 LLM 先生成一段"假设性答案"，用这段答案的向量去检索。

        原理：答案和文档的语义空间更近，比问题更容易匹配到相关文档。
        """
    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-turbo",
            "messages": [
                {"role": "system", "content": (
                    "根据用户问题，生成一段假设性的回答文档（100字左右），"
                    "风格像公司制度文档，语气正式。只输出文档内容。"
                )},
                {"role": "user", "content": query},
            ],
            "temperature": 0.5,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def hybrid_search_with_rewrite(collection, query: str,strategy: str = "rewrite", n_results: int=3,recall_n:int=5)->list[str]:
    """
    带 Query 改写的混合检索

    strategy:
        "none":     不改写
        "rewrite":  简单改写
        "expand":   问题扩展
        "hyde":     假设性文档
    混合检索：向量 + 关键词 → RRF 融合
    query → 向量检索 → 文档ID列表
    query → BM25 → 文档ID列表
    两个ID列表 → RRF融合 → 最终文档内容
    参数：
        collection: ChromaDB collection
        query: 用户问题
        n_results: 最终返回几条
        recall_n: 每路召回几条
    """
    # 向量检索
    # ===== 1. Query 改写 =====
    search_queries = [query]

    if strategy == "rewrite":
        rewritten = rewrite_query(query)
        print(f"  [改写] '{query}' → '{rewritten}'")
        search_queries = [rewritten]

    elif strategy == "expand":
        subs = expand_query(query, n=3)
        print(f"  [扩展] '{query}' → {subs}")
        search_queries = subs

    elif strategy == "hyde":
        hypothetical = hyde(query)
        print(f"  [HyDE] 假设文档: {hypothetical[:60]}...")
        # HyDE 只用向量检索（假设文档本身没有关键词意义）
        query_vec = get_embeddings([hypothetical])[0]
        results = collection.query(query_embeddings=[query_vec], n_results=recall_n)
        return results["documents"][0][:n_results]

    # ===== 2. 混合检索（对每个 search_query 分别检索）=====
    all_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_data["documents"]
    all_rank_lists = []
    for sq in search_queries:
        # 向量
        vec = get_embeddings([sq])[0]
        v_results = collection.query(query_embeddings=[vec], n_results=recall_n)
        v_ids = [meta['index'] for meta in v_results['metadatas'][0]]
        # 关键词
        bm25 = SimpleBM25(all_docs)
        k_ids = bm25.search(sq, top_k=recall_n)

        all_rank_lists.append(v_ids)
        all_rank_lists.append(k_ids)
        print(f"  [关键词检索] '{sq}' → {k_ids}")

    # 3. RRF 融合
    fused_ids = rrf_fusion(
        rank_lists=all_rank_lists,
        k=60,
        top_k=n_results,
    )
    print(f"  [RRF] 最终 top-{n_results}={fused_ids}")

    # 4. 返回文档内容
    return [all_docs[i] for i in fused_ids]


if __name__ == '__main__':
    print(f"\n向量库已加载（{collection.count()} 条记录）")

    test_questions = [
        ("RabbitMQ 用在什么场景？", "精确关键词"),
        ("咋报销", "口语化"),
        ("请假和报销的流程分别是什么？", "复合问题"),
        ("后端框架是什么？", "简单语义"),
    ]

    strategies = ["none", "rewrite", "expand", "hyde"]

    for q, tag in test_questions:
        print(f"\n{'='*55}")
        print(f"❓ {q}  [{tag}]")
        print(f"{'='*55}")

        for s in strategies:
            print(f"\n  【{s.upper()}】")
            try:
                docs = hybrid_search_with_rewrite(collection, q, strategy=s, n_results=2)
                # 只看检索结果，不调 LLM（省 token）
                for i, doc in enumerate(docs):
                    print(f"    #{i+1}: {doc[:50]}...")
            except Exception as e:
                print(f"    ❌ {e}")


# 实际生产策略
# 口语化特征词
COLLOQUIAL_PATTERNS = [
    # 疑问词
    r"咋", r"啥", r"咋整", r"咋办", r"咋搞", r"咋弄",
    r"搞啥", r"弄啥", r"整啥",
    # 口语动词
    r"怎么弄", r"怎么搞", r"怎么整", r"怎么弄",
    r"搞一下", r"弄一下", r"整一下",
    # 口语语气
    r"那个", r"这个那个", r"就是说",
    r"帮我看看", r"帮我看下",
    # 省略主语/对象
    r"^多少$", r"^什么时候$", r"^找谁",
]


def is_colloquial(query: str) -> bool:
    """
    检测用户问题是否为口语化表达

    返回 True → 需要改写
    返回 False → 直接检索
    """
    for pattern in COLLOQUIAL_PATTERNS:
        if re.search(pattern, query):
            return True
    return False
# 根据问题复杂度选择策略
"""
规则覆盖不了的场景（比如"年假能带明年吗"），生产环境可以加一层 LLM 判断兜底
"""
def choose_strategy(query: str) -> str:
    """根据问题特征选择检索策略"""
    # 复合问题
    if len(query) > 15 and any(kw in query for kw in ["和", "分别", "以及", "还有"]):
        return "expand"
    # 口语化
    if is_colloquial(query):
        return "rewrite"
    # 默认
    return "none"

tests = [
    "咋报销",              # True
    "报销怎么弄？",         # True
    "那个假怎么找谁批",     # True
    "报销流程是什么？",     # False
    "RabbitMQ 用在什么场景？", # False
    "3天以上的假需要谁审批？", # False
]

for q in tests:
    print(f"  {'🗣️' if is_colloquial(q) else '📝'} {q}")