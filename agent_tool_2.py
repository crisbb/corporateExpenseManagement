# 模型不执行工具，只决定"用哪个工具、传什么参数"。真正执行的是你的代码
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
@tool
def add(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个数的乘积"""
    return a * b

tool_map = {
    'add': add,
    'multiply': multiply
}
def get_llm():
    return ChatOpenAI(
        model="qwen-plus",
        api_key='sk-882d93e0832348979b1a3f1702bac021',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
# 绑定tool
llm_with_tools = get_llm().bind_tools([multiply, add])
messages = [HumanMessage(content='帮我算一下 123 乘以 456，再加上 789')]
response = llm_with_tools.invoke(messages)

messages.append(response)

if response.tool_calls:
    for tool_call in response.tool_calls:
        func = tool_map[tool_call['name']] # tool执行
        result = func.invoke(tool_call["args"])
        print(f"🔧 调用工具: {tool_call['name']}({tool_call['args']}) = {result}")
        messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call['id']
        ))

final_response = llm_with_tools.invoke(messages)
print(f"\n🤖 最终回答: {final_response.content}")

