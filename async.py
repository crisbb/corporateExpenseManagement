import asyncio

async def main():
    print('hello...')
    await asyncio.sleep(1)# 模拟网络延迟
    print('... World!')
# 运行异步主函数
asyncio.run(main())

# asyncio.gather
def read_file_safely(file_path):
    """
    尝试安全读取文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content
    except FileNotFoundError:
        # 专门捕获“文件未找到”的错误
        print(f'找不到{file_path}的文件')
        return None
    except PermissionError:
        # 额外捕获“权限不足”的错误（比如文件被其他程序占用）
        print(f"🚫 错误：没有权限读取文件 '{file_path}'。")
        return None
    
result = read_file_safely("non_existent_file.txt")
print(f"返回结果: {result}")