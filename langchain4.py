from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个{style}编程助手'),
    MessagesPlaceholder(variable_name="history"),
    ('human', '{question}')
])

llm = ChatOpenAI(
    model='qwen-plus',
    api_key='sk-882d93e0832348979b1a3f1702bac021',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
parser = StrOutputParser()
chain = prompt | llm | parser
store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
# 包装成带记忆的链
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='question',
    history_messages_key='history'
)
config = {'configurable': {'session_id': 'session_001'}}
print("=== 带记忆的对话（输入 quit 退出）===\n")

while True:
    user_input = input("你: ")
    if user_input.lower() =='quit':
        print("再见！")
        break
    response = chain_with_memory.invoke({'question': user_input,'style': '温柔'},config=config)
    print(f"AI: {response}\n")
