# 结构化输出
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 示例数据
examples = [
    ("今天心情超好！", "正面"),
    ("快递丢了，气死", "负面"),
    ("明天开会", "中性"),
    ("代码终于跑通了！", "正面"),
    ("又加班，烦死了", "负面"),
]
# 手动构建消息列表
messages = [
    ("system", "你是情感分类器，只输出：正面、负面、中性。")
]
for user_input, label in examples:
    messages.append(('user', user_input))
    messages.append(('ai', label))
# 最后加上真正的问题占位符
messages.append(('user', '{text}'))
prompt = ChatPromptTemplate.from_messages(messages)

llm = ChatOpenAI(
    model='qwen-plus',
    api_key='sk-882d93e0832348979b1a3f1702bac021',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

chain = prompt | llm | StrOutputParser()
# 调用
response = chain.invoke({ 'text': '你的裙子很漂亮'})
print(response)


