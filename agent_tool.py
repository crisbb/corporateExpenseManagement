# 模型不执行工具，只决定"用哪个工具、传什么参数"。真正执行的是你的代码
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
@tool
def add(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个数的乘积"""
    return a * b

def get_llm():
    return ChatOpenAI(
        model="qwen-plus",
        api_key='sk-882d93e0832348979b1a3f1702bac021',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

llm_with_tools = get_llm().bind_tools([multiply, add])

response = llm_with_tools.invoke("帮我算一下 123 乘以 456")
# 4. 查看模型的"决定"
print(response.content)           # 可能为空
print(response.tool_calls) #[{'name': 'multiply', 'args': {'a': 123, 'b': 456}, 'id': 'call_88667acfd4844ca697d475', 'type': 'tool_call'}]  没有结果 只有要调用的tool


