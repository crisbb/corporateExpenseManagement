# RAG求职冲刺学习计划 | 目标：AI应用工程师
**最后更新**：2026-08-27 | **关联仓库**：[crisbb/corporateExpenseManagement](https://github.com/crisbb/corporateExpenseManagement)

> 💡 **执行原则**：每天只做1个技术点 → 提交1个PR → 记录1个面试故事

## ✅ 每日执行模板
### Day 1 | 2026-08-27
**✅ 今日任务**：  
`实现响应安全处理器`  

## ✅ 每日执行模板
### Day 2 | 2026-08-28
**✅ 今日任务**：  
`rerank 精排`
`混合检索`

## ✅ 每日执行模板
### Day 3 | 2026-09-01
**✅ 今日任务**：  
`query改写`

**📚 学习内容**：
- `rewrite_query`：口语化 → 专业检索格式（如 "咋报销" → "报销流程是什么"）
- `expand_query`：复合问题拆成 N 个子问题，分别检索再合并
- `HyDE`：让 LLM 先生成假设性答案，用答案向量去检索（答案空间与文档更近）
- `自适应策略`：`is_colloquial` 规则检测口语化 → `choose_strategy` 自动选择 none/rewrite/expand

**🔧 代码提交**：  
```bash
git checkout -b feature/query-rewrite
git commit -m "[RAG] 实现 Query 改写：rewrite/expand/HyDE + 自适应策略选择"
git push
```

**🎯 面试故事**：  
> "我实现了三种 Query 改写策略：简单改写、问题扩展、HyDE 假设文档。
> 通过规则引擎（口语化特征词检测）自动选择策略：口语化问题走 rewrite，复合问题走 expand，简单问题直接检索。
> HyDE 的核心洞察是：用户问题和文档的语义空间差距大，但答案和文档的语义空间更接近，用假设答案的向量去检索能显著提升召回率。"