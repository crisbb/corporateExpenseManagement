# 结构化输出
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 定义输出结构
class CodeAnalysis(BaseModel):
    """代码分析结果"""
    language:str = Field(description='编程语言')
    function_name:str = Field(description='函数名')
    difficulty: str = Field(description="难度：简单/中等/困难")
    summary: str = Field(description="一句话总结功能")

llm = ChatOpenAI(
    model='qwen-plus',
    api_key='sk-882d93e0832348979b1a3f1702bac021',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
# 用 llm.with_structured_output绑定 Pydantic模型
structured_llm = llm.with_structured_output(CodeAnalysis)

# 调用
response = structured_llm.invoke('分析这段代码：def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)')
# 测试类型
print(f"响应类型: {type(response)}")
print(f"是否为 Pydantic 模型: {isinstance(response, BaseModel)}")

# 安全调用
if hasattr(response, "model_dump"):
    print(response.model_dump())
else:
    print("仍为字典类型，请检查 structured_llm 配置")

"""
类型安全 可以通过点语法访问
自动验证 按指定格式输出 否则报错
文档友好 des....作为提示词指导LLM生成
# 查看响应的所有属性
print(dir(response))

# 查看具体字段值     安全访问属性（推荐方式）
print(f"函数名称: {response.function_name}")
print(f"时间复杂度: {response.time_complexity}")

# 转换为字典（便于JSON序列化）
print(response.model_dump())

"""
