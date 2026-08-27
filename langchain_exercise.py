import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI 工具箱", page_icon="🧰")

# 初始化
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    st.header("⚙️ 设置")
    st.session_state.api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)

def get_llm():
    return ChatOpenAI(
        model="qwen-plus",
        api_key=st.session_state.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

st.title("🧰 AI 工具箱")

# 选择工具
tool = st.selectbox("选择工具", ["情感分析", "代码分析", "文本摘要"])

if not st.session_state.api_key:
    st.warning("请先输入 API Key")
else:
    # === 工具 1：情感分析（结构化输出）===
    if tool == "情感分析":
        class Sentiment(BaseModel):
            emotion: str = Field(description="情感：正面/负面/中性")
            confidence: str = Field(description="置信度：高/中/低")
            reason: str = Field(description="判断理由，一句话")

        text = st.text_area("输入文本：")
        if st.button("分析") and text:
            llm = get_llm()
            structured_llm = llm.with_structured_output(Sentiment)
            result = structured_llm.invoke(f"分析以下文本的情感：{text}")

            col1, col2, col3 = st.columns(3)
            col1.metric("情感", result.emotion)
            col2.metric("置信度", result.confidence)
            st.info(f"理由：{result.reason}")

    # === 工具 2：代码分析（结构化输出）===
    elif tool == "代码分析":
        class CodeInfo(BaseModel):
            language: str = Field(description="编程语言")
            purpose: str = Field(description="功能描述")
            bugs: list[str] = Field(description="潜在问题列表")
            suggestion: str = Field(description="改进建议")

        code = st.text_area("粘贴代码：")
        if st.button("分析") and code:
            llm = get_llm()
            structured_llm = llm.with_structured_output(CodeInfo)
            result = structured_llm.invoke(f"分析以下代码：\n{code}")

            st.write(f"**语言**：{result.language}")
            st.write(f"**功能**：{result.purpose}")
            if result.bugs:
                st.warning("⚠️ 潜在问题：")
                for bug in result.bugs:
                    st.write(f"- {bug}")
            else:
                st.success("✅ 未发现明显问题")
            st.info(f"💡 建议：{result.suggestion}")

    # === 工具 3：文本摘要（普通输出）===
    elif tool == "文本摘要":
        text = st.text_area("输入长文本：")
        max_words = st.slider("摘要字数限制", 20, 200, 50)

        if st.button("生成摘要") and text:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是摘要专家。用不超过{max_words}字总结内容，只输出摘要。"),
                ("human", "{text}")
            ])
            chain = prompt | get_llm() | StrOutputParser()
            result = chain.invoke({"text": text, "max_words": max_words})
            st.write(f"**摘要**：{result}")
