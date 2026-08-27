import streamlit as st

st.title('💬 AI 聊天助手')
# st.header("二级标题")
# st.write('hello')
# # === 文本展示 ===
# st.title("大标题")
# st.header("二级标题")
# st.write("普通文本，最常用")
# st.markdown("**支持 Markdown**，比如 `代码` 和 [链接](https://baidu.com)")

# === 用户输入 ===
text = st.text_input("输入你的问题", placeholder="请输入...")
# number = st.number_input("数字", min_value=0, max_value=100)
# slider = st.slider("滑块", 0, 100, 50)
# checkbox = st.checkbox("勾选我")
# select = st.selectbox("下拉选择", ["选项A", "选项B", "选项C"])

# # === 布局 ===
# col1, col2 = st.columns(2)
# with col1:
#     st.write("左边")
# with col2:
#     st.write("右边")

# === 侧边栏 ===
with st.sidebar:
    st.header("设置")
    api_key = st.text_input("API Key", type="password")

# name = st.text_input("你叫什么名字？")
# if name:
#     st.write(f'你好{name} 欢迎入坑')
# count = 0
# if 'count' not in st.session_state:
#     st.session_state.count = 0
# if st.button('点我+1'):
#     st.session_state.count += 1
# st.write(f"count = {st.session_state.count}")
# st.session_state 就是一个跨刷新的全局字典，用来保存状态。 后面做聊天机器人时，对话历史就存在这里面

if st.button('发送'):
    if(text):
        st.write(f'你说了 {text}')
    else:
        st.warning('请输入内容！')