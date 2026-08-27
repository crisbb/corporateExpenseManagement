# 结构化输出
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch


# 定义多个模板
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是翻译专家，将用户输入翻译成{language}。只输出翻译结果。"),
    ("human", "{text}")
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是摘要专家，用不超过50字总结以下内容。"),
    ("human", "{text}")
])

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是代码审查专家，找出以下代码的 bug 并给出修复。"),
    ("human", "{text}")
])

llm = ChatOpenAI(
    model='qwen-plus',
    api_key='sk-882d93e0832348979b1a3f1702bac021',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
# 用 RunnableBranch 创建动态链
chain = RunnableBranch(
    (lambda x : x.get('task')=='translate',translate_prompt | llm | StrOutputParser()),
    (lambda x: x.get('task')=='code', code_prompt | llm | StrOutputParser()),
    summarize_prompt | llm | StrOutputParser()
)
# 测试
print(chain.invoke({"task": "translate", "language": "英文", "text": "今天天气不错"}))
# → "The weather is nice today"

print(chain.invoke({"task": "summarize", "text": "Python是一种解释型语言...（长文本）"}))
# → 50字摘要

print(chain.invoke({"task": "code", "text": "def add(a,b): return a - b"}))


