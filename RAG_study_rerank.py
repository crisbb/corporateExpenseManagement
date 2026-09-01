# RAG 系统实现（基于阿里云百炼 + Chroma）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 【离线阶段】（只做一次）
#   文档 → 加载 → 分块 → 向量化 → 存入向量数据库

# 【在线阶段】（每次提问）
#   用户提问 → 向量化 → 在向量数据库中检索相似片段 → 塞进 prompt → 模型回答
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import EphemeralClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import httpx

LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = 'sk-882d93e0832348979b1a3f1702bac021'
# 加载文档 + 分块
loader = TextLoader('knowledge.txt', encoding='utf-8')
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

相关性排序（从高到低）："""

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

from langchain_core.documents import Document

# 自己加载文件，替代 TextLoader
def load_text(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content)]

# documents = load_text("knowledge.txt")

# ============ 3. 向量化（直接调阿里云 API）============

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

def get_embeddings(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "text-embedding-v3",
            "input": texts,  # 必须是字符串列表
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # 按 index 排序，保证顺序一致
    data["data"].sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data["data"]]

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

# ============ 5. 验证 ============
# 将问题向量化
# 计算与知识库片段的余弦相似度
# 返回 top-n 结果
def generate_rag_response(query: str, use_rerank: bool = True)-> str:
    """用户提问 → 检索 → (可选 Rerank) → LLM 回答"""
    # ===== 1. 多召回 =====
    query_vec = get_embeddings([query])[0] # 问题向量化
    recall_n = 5 if use_rerank else 2  # 多召回 5 条
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=recall_n # 返回最相似的2个片段
    )
    candidates = results["documents"][0]
    print(f"  [检索] 召回 {len(candidates)} 条候选")
   # ===== 2. Rerank 精排 =====
    if use_rerank and len(candidates)>2:
        context_docs = rerank_with_llm(query, candidates, top_k=2)
    else:
        context_docs = candidates[:2]
    context = '\n\n'.join(context_docs)

    # 构建增强提示词（关键增强！）
    prompt = ChatPromptTemplate.from_messages([
        ('system', "你是一个专业HR助手，严格基于以下公司文档回答问题。"
         "如果文档未提及，回答'根据公司现有资料无法确定'。"
         "回答要简洁，不要编造信息。"),
        ('human', '【公司文档】\n{context}\n\n【用户问题】\n{query}')
    ])
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        'context': context,
        'query': query
    })
    return response.content if hasattr(response, 'content') else str(response)

if __name__ == '__main__':
    print(f"\n向量库已加载（{collection.count()} 条记录）")

    test_questions = [
        "报销审批通过后多久打款？",
        "3天以上的假需要谁审批？",
        "后端用什么技术栈？",
        "公司用什么消息队列？",
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"❓ {q}")
        print(f"{'='*50}")
        
        # 无 Rerank
        print("\n  【无 Rerank】")
        ans1 = generate_rag_response(q, use_rerank=False)
        print(f"  💡 {ans1}")
        
        # 有 Rerank
        print("\n  【有 Rerank】")
        ans2 = generate_rag_response(q, use_rerank=True)
        print(f"  💡 {ans2}")


