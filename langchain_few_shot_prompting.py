# 结构化输出
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个情感分类器。根据用户输入，只输出一个词：正面、负面、中性。

示例：
输入：今天天气真好，心情愉快
输出：正面

输入：这个产品太垃圾了，退货！
输出：负面

输入：会议改到下午三点
输出：中性"""),
    ('human', '{text}')
])

llm = ChatOpenAI(
    model='qwen-plus',
    api_key='sk-882d93e0832348979b1a3f1702bac021',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

chain = prompt | llm | StrOutputParser()
# 调用
response = chain.invoke({ 'text': '这家菜味道还行，服务一般'})
print(response)


