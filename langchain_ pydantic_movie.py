# 结构化输出
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 定义输出结构
class MovieReview(BaseModel):
    title:str = Field(description='标题')
    rating:int = Field(description='评分0-10')
    pors:list[str] = Field(description='优点列表')
    cons:list[str] = Field(description='缺点列表')
    one_line: str = Field(description="一句话影评")


llm = ChatOpenAI(
    model='qwen-plus',
    api_key='sk-882d93e0832348979b1a3f1702bac021',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业影评人，请结构化评价用户提到的电影。"),
    ('human', '请评价电影{movie}')
])
# !!!创建 structured_llm 后，必须确保它在链的最后一步，否则结构化处理不会触发！
# 用 llm.with_structured_output绑定 Pydantic模型
structured_llm = llm.with_structured_output(MovieReview)
# 告诉模型输出 JSON → 解析 JSON → 验证字段 → 转成 Pydantic 对象
chain = prompt | structured_llm
# 调用
response = chain.invoke({ 'movie': '美人鱼'})
# 测试类型
print(f"响应类型: {type(response)}")
print(f"是否为 Pydantic 模型: {isinstance(response, BaseModel)}")

# 安全调用
if hasattr(response, "model_dump"):
    print(response.model_dump()) # 转成字典
else:
    print("仍为字典类型，请检查 structured_llm 配置")

