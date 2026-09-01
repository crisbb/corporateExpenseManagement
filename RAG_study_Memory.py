# RAG 系统实现（基于阿里云百炼 + Chroma）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dataclasses import dataclass, field
from langchain_community.document_loaders import TextLoader # 文本加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter #智能分块器
from chromadb import EphemeralClient # 内存版ChromaDB(重启丢失)
from collections import Counter # 词频计数器


"""
多轮对话 RAG 关键：                                 │
│                                                     │
│  1. 检索用"改写后的独立问题"（消除指代）              │
│  2. 生成用"原始问题 + 历史"（让 LLM 理解语境）       │
│  3. 滑动窗口限制历史长度（避免 Prompt 过长）          │
│                                                     │
│  为什么检索和生成用不同的信息？                       │
│  检索 → 需要精准关键词 → 用改写后的完整问题          │
│  生成 → 需要理解语境 → 用原始问题 + 历史对话 
"""

@dataclass
class ChatMessage:
    role: str
    content: str
class ConversationMemory:
    def __init__(self, max_turns: int = 3):
        self.messages: list[ChatMessage] = []
        self.max_turns = max_turns

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role, content))
        while len(self.messages) > self.max_turns:
            self.messages.pop(0)
    def get_messages(self) -> str:
        lines = []
        for msg in self.messages:
            prefix = "用户" if msg.role == "user" else "AI"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
    def clear(self):
        self.messages.clear()


def rewrite_with_context(query: str, history: str) -> str:
    """
    结合对话历史，将用户的后续问题改写为独立完整的问题。

    例：
        历史: 用户: 报销怎么打款？ AI: 3个工作日...
        当前: 那请假呢？
        改写: 请假的审批流程是什么？
    """
    prompt = f"""根据对话历史，将用户的最新问题改写为一个独立、完整、可检索的问题。
要求：
- 消除指代词（"那个""它""上面的"等）
- 补充上下文中的隐含信息
- 只输出改写后的问题，不要解释

对话历史：
{history}

用户最新问题：{query}

改写后的独立问题："""

    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# 【离线阶段】（只做一次）
#   文档 → 加载 → 分块 → 向量化 → 存入向量数据库

# 【在线阶段】（每次提问）
#   用户提问 → 向量化 → 在向量数据库中检索相似片段 → 塞进 prompt → 模型回答


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

def rag_chat_stream(collection, query: str, memory: ConversationMemory)->str:
    """
        多轮对话 RAG：
        1. 结合历史改写问题
        2. 混合检索
        3. 调 LLM 回答（带上历史上下文）
        4. 存入记忆
        """
    # 1. 改写（如果有历史）
    if memory.messages:
        history = memory.get_messages()
        rewritten = rewrite_with_context(query, history)
        print(f"  [改写] '{query}' → '{rewritten}'")
    else:
        rewritten = query
        print(f"  [改写] 首轮对话，无需改写")

    # ===== 2. 混合检索（对每个 search_query 分别检索）=====
    # 2. 检索
    all_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_data["documents"]

    # 向量
    query_vec = get_embeddings([rewritten])[0]
    v_results = collection.query(query_embeddings=[query_vec], n_results=5)
    v_ids = [meta["index"] for meta in v_results["metadatas"][0]]

    # 关键词
    bm25 = SimpleBM25(all_docs)
    k_ids = bm25.search(rewritten, top_k=5)

    # RRF
    fused_ids = rrf_fusion(rank_lists=[v_ids, k_ids], k=60, top_k=2)
    context_docs = [all_docs[i] for i in fused_ids]
    context = "\n\n".join(context_docs)
    print(f"  [检索] top-2={fused_ids}")


    # 3. 调 LLM（带上历史）
    history_for_llm = memory.get_messages() if memory.messages else "（无）"
    system_prompt = (
        "你是一个专业HR助手，严格基于公司文档回答用户问题。\n"
        "如果文档未提及，回答'根据公司现有资料无法确定'。\n"
        "回答要简洁，不要编造信息。"
    )

    user_prompt = f"""【对话历史】
    {history_for_llm}

    【公司文档】
    {context}

    【用户问题】
    {query}"""

    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-plus",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "stream": True
        },
        timeout=60,
    )
    resp.raise_for_status()
    # 逐行解析 SSE
    # 逐行解析 SSE
    full_answer = ""
    for line in resp.iter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            import json
            data = json.loads(data_str)
            delta = data["choices"][0]["delta"].get("content", "")
            if delta:
                full_answer += delta
                yield delta
    # 4. 存入记忆
    memory.add_message("user", query)
    memory.add_message("assistant", full_answer)


if __name__ == '__main__':
    print(f"\n向量库已加载（{collection.count()} 条记录）")

    memory = ConversationMemory(max_turns=3)

    test_questions = [
        "报销怎么打款？",
        "那请假呢？",
        "后端用什么框架？",
    ]

    for q in test_questions:
        print(f"\n{'='*55}")
        print(f"❓ {q}")
        print(f"{'='*55}")

        # 流式输出：逐字打印
        print("💡 ", end="", flush=True)
        for token in rag_chat_stream(collection, q, memory):
            print(token, end="", flush=True)
        print()  # 换行

    # 问题在于
    # httpx.post()
    # 是同步请求，它会等整个响应全部接收完才返回，所以
    # iter_lines()
    # 只是在遍历已经缓冲好的内容 —— 并不是真正的实时流式。