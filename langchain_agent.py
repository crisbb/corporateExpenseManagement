from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# AgentExecutor 的本质 = 一个 while 循环：
#   1. 调模型
#   2. 模型说要调工具 → 执行工具 → 把结果加回 messages → 回到第 1 步
#   3. 模型不调工具了 → 输出最终回答 → 结束

# 1. 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """计算两个数的乘积"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b

@tool
def get_word_length(word: str) -> int:
    """获取一个单词的字母数量"""
    return len(word)
@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    import datetime
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

tools = [multiply, add, get_word_length, get_current_time]
tool_map = { k.name: k for k in tools}

# 2. 创建模型
llm = ChatOpenAI(
    model="qwen-plus",
    api_key='sk-882d93e0832348979b1a3f1702bac021',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
llm_with_tools = llm.bind_tools(tools)
# 4. 循环执行（替代 AgentExecutor）
def run_agent(question: str, max_iterations: int = 5):
    messages = [HumanMessage(content=question)]
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content
        
        for tool_call in response.tool_calls: # 只负责执行工具
            func = tool_map[tool_call['name']]
            result = func.invoke(tool_call['args'])
            print(f"  🔧 {tool_call['name']}({tool_call['args']}) → {result}")

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return "达到最大迭代次数，未能得出答案"

# 5. 测试
print(run_agent("帮我算一下 123 乘以 456，再加上 789"))
print(run_agent("现在几点了？"))
print(run_agent("什么是递归？")) 

# 场景	模型行为
# 多个工具之间无依赖	一次返回多个 tool_calls
# 工具之间有依赖（后一个需要前一个的结果）	分多轮，每轮调一个
# 不需要工具	返回空，直接回答
# 装饰器里的描述非常重要
