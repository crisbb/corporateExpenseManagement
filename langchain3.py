from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个{style}编程助手'),
    ('human', '{question}')
])

llm = ChatOpenAI(
    model='qwen-plus',
    api_key='sk-882d93e0832348979b1a3f1702bac021',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
parser = StrOutputParser()
chain = prompt | llm | parser

# 用 .stream() 代替 .invoke()
print('AI', end='')
for chunk in chain.stream({'question': '用三句话解释什么是 RAG', 'style': '热情'}):
    print(chunk, end="", flush=True)
print()
# AIRAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与大语言模型生成能力相结合的AI技术。  
# 它先从外部知识库（如文档、数据库或网页）中实时检索与用户问题最相关的片段，再将这些片段作为上下文输入给大语言模型，辅助其生成更准确、可信且有依据的回答。  
# 相比纯参数化模型，RAG能动态利用最新/专有数据，缓解幻觉问题，并支持可追溯、可更新的知识使用方式。