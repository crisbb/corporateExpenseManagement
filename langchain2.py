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


response = chain.invoke({
    'style': '温柔耐心',
    'question': '列表推导式和普通 for 循环有什么区别？'
})
print(response)