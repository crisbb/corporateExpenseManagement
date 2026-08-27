"""
项目名称：AI 文本处理小助手
作者：你的名字
日期：2023-10-27
描述：这是一个用于演示 Python 注释规范的脚本。
"""


name = 'Tom'
age = 25
is_active = True

greeting = f'hello,{name},age{age}'

fruits = ['apple','banana']
fruits.append('cherry')
fruits[0]
len(fruits)

nums = [1,2,3,4,5]
doubles = [n*2 for n in nums]
evens = [n for n in nums if n % 2 == 0]

user = {'name':'Tom','age': 25}
user['name']
user.get('email', 'none')
user['email'] = 'a@163@com'

res = None

def add(a: int, b: int = 0) -> int:
    return a + b

if age >=18:
    print('adult')
elif age >=12:
    print('teen')
else:
    print('kid')

for fruit in fruits:
    print(fruit)

for i, fruit in enumerate(fruits):
    print(i, fruit)

label = 'adult' if age >=18 else 'kid'

class User:
    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f'Hi,{self.name}'
    
u = User('Tom')
print(u.greet())

"""
这里
是
多行
注释
"""
print("""
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""")

print(3 * 'un' + 'ium')

# 相邻字符串会合并
print('wq' '121s')

def call_llm_api(prompt, temperature=0.7):
    """
    调用大模型 API 的核心函数。
    
    参数:
        prompt (str): 发送给大模型的提示词。
        temperature (float): 控制输出随机性的参数，默认 0.7。
        
    返回:
        str: 大模型生成的文本回复。
    """
    # 这里是具体的代码逻辑...
    return "AI的回复"

word = '{{Python}}'
print(word[0]) # 0 号位的字符 -0 和 0 一样，因此，负数索引从 -1 开始。
print(word[-0])
print(word[5])  # 5 号位的字符

print(word[-1])  # 倒数第一位的

print(word[1:2]) # 不含2 省略开始索引时，默认值为 0，省略结束索引时，默认为到字符串的结尾
# 输出结果包含切片开始，但不包含切片结束。因此，s[:i] + s[i:] 总是等于 s：

print(f'{word}{{...}}') #任何在替换字段之外的双花括号 ({{ 或 }}) 都将以相应的单花括号替换:


"""
替换字段之外的其他字符将被当作是普通的字符串字面值。
这意味着转义序列会被解码（除非该字面值还被标记为原始字符串），
在三重引号 f-字符串中还可以换行
"""
name = 'Galahad'
favorite_color = 'blue'
print(f'{name}:\t{favorite_color}')

print(rf"C:\Users\{name}")

print(f'''Three shall be the number of the counting
and the number of the counting shall be three.''')

print(f'{(a := 1/2)}, {a * 42}')# 每个表达式都将在格式化字符串字面值所在的上下文中以从左至右的顺序被求值。 空表达式不被允许，而 lambda 和赋值表达式 := 必须显式地以圆括号标记

# 允许在替换字段中重用外层 f-字符串的引号类型
dic = dict(a = 2)
print(f'as {dic['a']} as')

# 分隔符字符串.join(可迭代对象)
print(f'{'-'.join(['a','b','c'])}')
# 如果你用的是最新的 Python 3.12，原代码不会报错，但为了代码兼容性，建议养成内外引号错开的习惯
print(f'List a contains:\n{'\n'.join(['aa','bb'])}')

# 嵌套
name = 'world'
f'Repeated:{f' hello {name}' * 3}'

# 在单个引号和三重引号 f-字符串中的替换表达式都可包含换行并可包含注释。 替换字段内部在 # 之后的任何内容都属于注释（包括花括号和引号）。 这意味着带有注释的替换字段必须另起一行来结束
fff = 2
print(f"abc{fff  # 在此行结束前此注释  }"  将保持有效
+ 3}")

letters = [1,2,3,4,5,6]
letters[2:5] = [8,8,8]
print(letters) # 会修改原数组
letters[:] = [] # 清空原数组
letters.clear() # 清空原数组
del letters[:] # 清空原数组

letters = [1,2,3,4,5,6]
new_list = letters[:]
new_list[:2] = [9]
print(new_list)
print(len(new_list))

# 嵌套列表
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
print(x)

# 斐波那契数列：
# 前两项之和即下一项的值 元组
a, b = 0, 1
while a < 10:
    print(a)
    (a, b) = (b, a+b)

a, b = 0, 1
while a < 1000:
    print(a, end='\n')
    a, b = b, a+b

height = 100
print(height // 3) # 向下取整

# strip 去除指定字符
'   spacious   '.strip()

'www.example.com'.strip('cmowz.')

#split 指定字符分割
'1,2,3'.split(',')

'1,2,3'.split(',', maxsplit=1)

'1,2,,3,'.split(',')

'1<>2<>3<4'.split('<>')

#join 数组 
', '.join(['spam', 'spam', 'spam'])

'-'.join('Python') # 'P-y-t-h-o-n'

"""
语法：list[start:stop:step]

start: 开始索引（包含）。
stop: 结束索引（不包含）。
step: 步长（默认为 1）。
"""
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
data[:3] # 前3个
data[-3:] # 最后3个

print(data[::-1]) # 不改变原数据
print(data)

# 6. 每隔一个取一个 (步长为 2)
print(data[::2])

my_list = [1,2,3,4]
#末尾添加
my_list.append(5)

print(my_list)
# 在指定位置插入
my_list.insert(2, 0)
print(my_list)

# pop(): 移除并返回最后一个元素 (像栈一样)
last_item = my_list.pop() 
print(last_item)

# remove(): 移除第一个匹配的值
my_list.remove(1)

# clear(): 清空列表

nums = [10, 20, 30, 20, 40]

# 6. index(): 找到某个值第一次出现的索引
print(nums.index(20))   # 输出: 1 (注意：如果找不到会报错，建议先用 in 判断)

# 7. count(): 统计某个值出现的次数
print(nums.count(20))   # 输出: 2

# 8. in 关键字: 判断是否存在 (非常重要！)
if 30 in nums:
    print("找到了！")

words = ["banana", "apple", "cherry"]

# 9. sort(): 原地排序 (直接修改原列表)
words.sort() 
# 结果: ["apple", "banana", "cherry"]

# 10. reverse(): 原地反转
words.reverse()
print(words)
# 结果: ["cherry", "banana", "apple"]

# 11. sorted(): 返回一个新排序的列表 (不修改原列表)
words1 = ["banana", "apple", "cherry"]
new_words = sorted(words1)
print(words1)

#在循环中删除元素
my_list = [item for item in my_list if item != 2]
print(my_list)

"""
控制流
"""

x = int(input("Please enter an integer: "))

if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')

users = {'Hans': 'active', 'Éléonore': 'inactive', '景太郎': 'active'}
for user,status in users.copy().items(): # 创建一个遍历副本
    if status == 'inactive':
        del users[user]

new_users = {} # 创建一个新多项集
for user, status in users.items():
    if (status == 'active'):
        new_users[user] = status

# range() 函数 用于生成等差数列 对象称为可迭代对象 iterable
for i in range(5):
    print(i)
# list(range(5, 10))
# [5,6,7,8,9]

# list(range(0, 10, 3))
# [0,3,6,9]

# list(range(-10, -100, -30))
# [-10,-40,-70]
sum(range(5))# 以可迭代对象作为参数的函数例如 sum()

# enumerate() 返回的迭代器的 __next__() 方法返回一个元组，里面包含一个计数值（从 start 开始，默认为 0）和通过迭代 iterable 获得的值。
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
print(list(enumerate(seasons)))
# [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
print(list(enumerate(seasons, start=1)))
# [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]

# break 跳出最近一层的for/while循环
# continue 语句将继续执行循环的下一次迭代


# 在 for 循环中，else 子句会在循环结束其最后一次迭代之后，即未执行 break 的情况下被执行
for n in range(2, 10):
    print(n)
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        # 循环到底未找到一个因数
        print(n, 'is a prime number')

# match语句
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case _:
            return "Something's wrong with the internet"

point = (0,0)
# point 是一个 (x, y) 元组
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y={y}")
    case (x, 0):
        print(f"X={x}")
    case (x, y):
        print(f"X={x}, Y={y}")
    case _:
        raise ValueError("Not a point")

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def where_is(point):
    match point:
        case Point(x=0, y=0):
            print("Origin")
        case Point(x=0, y=y):
            print(f"Y={y}")
        case Point(x=x, y=0):
            print(f"X={x}")
        case Point():
            print("Somewhere else")
        case _:
            print("Not a point")

"""
定义函数
"""
def fib2(n):  # 返回斐波那契数组直到 n
    """Return a list containing the Fibonacci series up to n."""
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)    # 见下
        a, b = b, a+b
    return result

f100 = fib2(100)    # 调用它
print(f100)

# 默认参数值 和 关键字参数
"""
纯位置	func(1, 2)	✅	按顺序匹配，简单粗暴。
纯关键字	func(b=2, a=1)	✅	名字匹配，顺序无关，最推荐。
混合（正序）	func(1, b=2)	✅	位置在前，关键字在后。
"""
# 解包实参列表 解包操作符
list(range(3,6))
args = [3,6]
list(range(*args))
"""
*	单星号	列表/元组	func(*[1, 2]) 变成 func(1, 2)
**  双星号  字典        func(**{"a":1, "b":2}) 变成 func(a=1, b=2)
"""

def my_func(x, y, z):
    return x + y + z

# 准备数据
my_list = [1, 2, 3]
my_dict = {"x": 10, "y": 20, "z": 30}

# 使用解包
print(my_func(*my_list))   # 相当于 my_func(1, 2, 3) -> 输出 6
print(my_func(**my_dict))  # 相当于 my_func(x=10, y=20, z=30) -> 输出 60



i = 5

def f(arg=i):
    print(arg)

i = 6
f() # 默认值在 定义 作用域里的函数定义中求值 5

"""
sorted(可迭代对象, key=排序规则, reverse=是否反转)
sorted()：不修改原列表，返回一个新的排好序的列表。
key：是一个“转换器”。它告诉 Python：“别直接比字典，先把我指定的这个字段拿出来，比这个字段！”
lambda：只是一个临时的、简单的“提取工具”。
"""
students = [
    {"name": "小明", "age": 18},
    {"name": "小红", "age": 16},
    {"name": "小刚", "age": 20}
]
sorted_students = sorted(students, key = lambda s:s['age'], reverse=True)
print(sorted_students)

# extend合并两个列表
a = [1, 2, 3]
b = [4, 5, 6]

a.extend(b) 
print(a)  # [1, 2, 3, 4, 5, 6]

all_docs = []

# 第一批检索结果
batch_1 = ["文档A", "文档B"]
all_docs.extend(batch_1) 

# 第二批检索结果
batch_2 = ["文档C", "文档D"]
all_docs.extend(batch_2)

print(all_docs) 
# ['文档A', '文档B', '文档C', '文档D']


# list.copy()
# 返回列表的浅拷贝。 类似于 a[:]
# insert, remove 或 sort 等仅修改列表的方法都不会打印返回值 -- 它们返回默认值 None

#列表推导式
squares = list(map(lambda x: x**2, range(10)))
print(squares)
squares2 = [x**2 for x in range(10)]
print(squares2)
# 以下列表推导式将两个列表中不相等的元素组合起来
print([(x,y) for x in [1,2,3] for y in [1,3,4] if x!= y])

docs = [
    {"text": "A", "score": 0.9},
    {"text": "B", "score": 0.5},
    {"text": "C", "score": 0.95}
]

# 一行代码搞定过滤和提取
high_quality_texts = [doc['text'] for doc in docs if doc['score']>0.8]
# 结果: ['A', 'C']
#嵌套推导 记忆口诀：“先写结果，再写内层循环，最后写外层循环。”
# 外层循环：for i in range(4)
# 作用：决定新矩阵有几行（也就是原矩阵有几列）。
# 内层推导式：[row[i] for row in matrix]
# 作用：生成新矩阵的某一行的具体内容。
# 它遍历原矩阵的每一行（row），取出该行在索引 i 位置的元素。
matrix = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
]
new_matrix = [[row[i] for row in matrix] for i in range(4)]
print(new_matrix)

# 元组 (Tuples) 和 序列 (Sequences)
# 用途：
# 保护数据：确保某些配置参数不被意外修改。
# 字典的键：只有不可变对象才能做字典的键。
# 函数返回多个值：return a, b 其实返回的是一个元组
# 处理、交换变量
first, *middle, last = [1, 2, 3, 4, 5]

print(first)   # 1
print(middle)  # [2, 3, 4]  <-- 注意：星号变量拿到的是一个列表
print(last)    # 5

# 去重Sets
ids = [101,102,103,104,101]
print(set(ids))
strategy_a = {1, 2, 3, 4}
strategy_b = {3, 4, 5, 6}
common_docs = strategy_a & strategy_b 

# 结果: {3, 4}
# dict() 构造函数
dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
dict(sape=4139, guido=4127, jack=4098)
{x : x**2 for x in (2,4,6)}

knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for (k,v) in knights.items():
    print(k,v)

names = ['tic', 'tac', 'toe']
for (i,v) in enumerate(names):
    print(i,v)
questions = ['a','b','c']
answers = ['n','p','q']
for (q,a) in zip(questions,answers):
    print('my name is {0} It is {1}'.format(q, a))
# my name is a It is n
# my name is b It is p
# my name is c It is q

for i in reversed(range(0, 10, 2)):
    print(i)
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for i in sorted(set(basket)):
    print(i)
print(basket)