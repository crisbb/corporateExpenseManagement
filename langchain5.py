import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="LangChain 聊天", page_icon="🦜", layout="wide")
if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("API Key", type="password")

    style = st.selectbox("助手风格", ["专业严谨", "幽默风趣", "毒舌但有用", "温柔耐心"])

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()
def build_chain(api_key: str):

    prompt = ChatPromptTemplate.from_messages([
        ('system', '你是一个{style}编程助手'),
        MessagesPlaceholder(variable_name="history"),
        ('human', '{question}')
    ])

    llm = ChatOpenAI(
        model='qwen-plus',
        api_key=api_key,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        streaming=True
    )
    return prompt | llm | StrOutputParser()

# === 主区域 ===
st.title("🦜 LangChain 聊天助手")
# 渲染历史
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if user_input := st.chat_input('输入你的问题...'):
    if not api_key:
        st.warning("⚠️ 请先在侧边栏输入 API Key")
    else:
        # 保存用户消息
        st.session_state.messages.append({'role': 'user','content': user_input})
        with st.chat_message('user'):
            st.write(user_input)
        # 构建历史（LangChain 格式）
        history = []
        for msg in st.session_state.messages[:-1]:# 排除刚加的这条
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        # 调用
        try:
            chain = build_chain(api_key)
            with st.chat_message('assistant'):
                placeholder = st.empty()
                full_content = ""
                for chunk in chain.stream({
                    'style': style,
                    'question': user_input,
                    'history': history
                }):
                    full_content+=chunk
                    placeholder.write(full_content + "▌")
                placeholder.write(full_content)
            st.session_state.messages.append({"role": "assistant", "content": full_content})
        except Exception as e:
                st.error(f"❌ 出错: {e}")
                st.session_state.messages.pop()



