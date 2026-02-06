# insights_prompt = {
#     "zh":"""
# 你是一位专业的信息检索领域专家，请基于以下一周内的论文数据，生成3-5条核心洞察。
#
# 论文数据：
# {papers_text}
#
# 请分析这些论文，总结出本周信息检索领域的主要趋势和核心洞察。每条洞察应该：
# 1. 简洁明了，不超过100字
# 2. 基于实际论文内容
# 3. 突出重要趋势或发现
# 4. 具有实际价值
#
# 请严格按照以下JSON格式返回：
# {{
#     "insights": [
#         "洞察1",
#         "洞察2",
#         "洞察3",
#         "洞察4",
#         "洞察5"
#     ]
# }}
#
# 注意：
# - 确保返回有效的JSON格式
# - 洞察数量可以是3-5条
# - 每条洞察应该独立且有意义
# /no_think
# """,
#     "en":"""
# You are a professional expert in the information retrieval field. Please generate 3-5 key insights based on the following week's paper data.
#
# Paper Data:
# {papers_text}
#
# Please analyze these papers and summarize the main trends and key insights in the information retrieval field this week. Each insight should:
# 1. Be concise and clear, no more than 100 words
# 2. Be based on actual paper content
# 3. Highlight important trends or findings
# 4. Have practical value
#
# Please return in the following JSON format:
# {{
#     "insights": [
#         "Insight 1",
#         "Insight 2",
#         "Insight 3",
#         "Insight 4",
#         "Insight 5"
#     ]
# }}
#
# Note:
# - Ensure valid JSON format
# - Number of insights can be 3-5
# - Each insight should be independent and meaningful
# - English
# /no_think
# """
# }
insights_prompt = {
    "zh": """
你通过分析最新的 ArXiv 论文数据，为信息检索（IR）领域的高级研究员撰写一份技术情报周报。

你的任务是从以下论文数据中，提炼出 3-5 个具有**高技术含量的核心洞察**。

**论文数据：**
{papers_text}

**"核心洞察"的撰写标准（必须严格遵守）：**
1. **拒绝宏观废话**：严禁使用"成为热点"、"广泛应用"、"显著提升"等空泛描述。
2. **聚焦具体方法论**：必须指明具体的技术手段（如：是使用了对比学习、稀疏检索、还是特定的注意力机制优化？）。
3. **问题-方案-效果**：每个洞察应遵循"针对什么具体难题 -> 采用了什么特定技术 -> 达到了什么效果/发现了什么反直觉结论"的逻辑。
4. **寻找技术共性**：如果多篇论文涉及同一主题（如RAG），不要只说RAG很火，而要总结它们在RAG的哪个具体环节（索引、召回、生成）做了改进。

**反面示例（不要这样做）：**
-  "多模态检索技术持续突破，在多个领域得到应用。" (太泛)
-  "推荐系统正在融合LLMs，提升了推荐的多样性。" (无技术细节)

**正面示例（请参考这种深度）：**
- "针对RAG长上下文的检索延迟问题，本周多篇论文转向**分层索引与推测性解码（Speculative Decoding）**结合的方案，在保持精度的同时将推理速度提升了3倍。"
- "在推荐系统中，生成式检索（Generative Retrieval）开始引入扩散模型来解决冷启动时的ID映射坍塌问题，比传统对比学习方法具有更好的鲁棒性。"

请严格按照以下 JSON 格式返回：
{{
    "insights": [
        "洞察内容1",
        "洞察内容2",
        "洞察内容3",
        "洞察内容4",
        "洞察内容5"
    ]
}}

注意：
- 确保返回有效的 JSON 格式
- 洞察数量为 3-5 条
- 每条洞察字数控制在 80-120 字之间，言之有物
/no_think
""",
    "en": """
You are acting as a Lead Research Analyst writing a technical intelligence report for senior researchers in Information Retrieval (IR).

Your task is to synthesize 3-5 **high-density technical insights** from the following paper data.

**Paper Data:**
{papers_text}

**Criteria for "Key Insights" (Strictly Enforced):**
1. **No Fluff**: Strictly forbid generic phrases like "is becoming popular", "widely applied", or "significant progress".
2. **Focus on Methodology**: You must specify the *technical approach* (e.g., Contrastive Learning, Late Interaction, KV Cache optimization, etc.).
3. **Problem-Solution-Outcome**: Structure each insight as: "Specific Problem -> Specific Technical Solution -> Concrete Outcome/Counter-intuitive Finding".
4. **Identify Patterns**: Do not just list topics. If multiple papers discuss RAG, summarize *what specific component* of RAG (indexing, retrieval, generation) is being optimized this week.

**Negative Examples (DO NOT DO THIS):**
- "Multimodal retrieval is breaking through and applied in many fields." (Too generic)
- "Recommender systems are integrating LLMs to improve diversity." (Lacks technical detail)

**Positive Examples (AIM FOR THIS DEPTH):**
- "To address retrieval latency in long-context RAG, recent papers are shifting towards Hierarchical Indexing combined with Speculative Decoding, achieving a 3x speedup while maintaining recall."
- "In Generative Retrieval, researchers are adopting Diffusion Models** to mitigate ID mapping collapse** during cold starts, outperforming traditional contrastive methods in robustness."

Please return in the following JSON format:
{{
    "insights": [
        "Insight 1",
        "Insight 2",
        "Insight 3",
        "Insight 4",
        "Insight 5"
    ]
}}

Note:
- Ensure valid JSON format
- Provide 3-5 insights
- Each insight should be dense and technical (80-120 words)
- English
/no_think
"""
}

keywords_prompt = {
    "zh":"""
你是一位专业的信息检索领域专家，请基于以下一周内的论文内容，提取5-8个新兴关键词。

论文内容摘要：
{combined_text}

请分析这些论文内容，提取出：
1. 新兴的技术术语
2. 热门研究方向
3. 重要的方法或概念
4. 新兴的应用领域

请严格按照以下JSON格式返回：
{{
    "keywords": [
        "关键词1",
        "关键词2",
        "关键词3",
        "关键词4",
        "关键词5",
        "关键词6",
        "关键词7",
        "关键词8"
    ]
}}

注意：
- 确保返回有效的JSON格式
- 关键词数量可以是5-8个
- 关键词应该是新兴的、有意义的术语
- 避免过于通用的词汇
/no_think
""",
    "en":"""
You are a professional expert in the information retrieval field. Please extract 5-8 emerging keywords based on the following week's paper content.

Paper Content Summary:
{combined_text}

Please analyze this paper content and extract:
1. Emerging technical terms
2. Hot research directions
3. Important methods or concepts
4. Emerging application areas

Please return in the following JSON format:
{{
    "keywords": [
        "Keyword 1",
        "Keyword 2",
        "Keyword 3",
        "Keyword 4",
        "Keyword 5",
        "Keyword 6",
        "Keyword 7",
        "Keyword 8"
    ]
}}

Note:
- Ensure valid JSON format
- Number of keywords can be 5-8
- Keywords should be emerging and meaningful terms
- Avoid overly generic words
/no_think
"""
}

title_prompt = {
    "zh": """
你是一位资深的信息检索领域编辑，擅长从学术论文中提炼核心创新点并生成吸引眼球的标题。

论文数据：
{papers_text}

周报信息：
- 周数：第{week_num}周
- 日期范围：{date_range}

任务步骤：
1. **深度分析**：仔细阅读所有论文，识别：
   - 本周最具突破性的技术创新（具体技术名称，如Transformer变体、知识蒸馏、强化学习等）
   - 新兴应用场景（如代码检索、医疗问答、跨语言搜索等）
   - 独特的方法论或架构设计
   - 解决的关键痛点或挑战

2. **差异化要求**：
   - 避免使用常见高频词：多模态、检索增强、生成AI、大语言模型、推荐系统
   - 如果必须提及这些概念，请具体化（如：视觉-文本对齐、分层检索架构、个性化生成等）
   - 挖掘本周论文的独特角度，与前几周形成鲜明对比

3. **标题风格**（选择最合适的一种）：
   - 技术突破型：强调具体的新技术、新方法（如"Mamba架构重塑序列建模"）
   - 应用创新型：突出新应用场景或领域（如"从代码到法律：垂直领域检索新范式"）
   - 问题解决型：聚焦解决的关键挑战（如"攻克长文档理解的效率瓶颈"）
   - 趋势洞察型：揭示研究方向转变（如"从规模竞赛到效率优先"）

4. **标题要求**：
   - 格式：第{week_num}周: [核心主题]
   - 长度：核心主题15-25字
   - 必须包含至少1个具体技术术语或应用场景
   - 要有"标题党"的吸引力，但不失专业性
   - 让读者一眼看出本周与众不同之处

请严格按照以下JSON格式返回：
{{
    "title": "第{week_num}周: [核心主题]",
    "reasoning": "简述为什么选择这个标题（1-2句话）"
}}

示例（仅供参考风格，不要抄袭）：
- "第X周: Sparse Attention机制引领百万Token处理新时代"
- "第X周: 零样本跨模态检索突破语言边界"
- "第X周: 从Prompt工程到架构创新的范式转移"

/no_think
""",

    "en": """
You are a senior editor in the information retrieval field, skilled at extracting core innovations from academic papers and generating eye-catching titles.

Paper Data:
{papers_text}

Weekly Report Information:
- Week: Week {week_num}
- Date Range: {date_range}

Task Steps:
1. **Deep Analysis**: Carefully read all papers and identify:
   - Most breakthrough technical innovations this week (specific tech names, e.g., Transformer variants, knowledge distillation, RL)
   - Emerging application scenarios (e.g., code retrieval, medical QA, cross-lingual search)
   - Unique methodologies or architectural designs
   - Key pain points or challenges addressed

2. **Differentiation Requirements**:
   - Avoid common high-frequency terms: multimodal, retrieval-augmented, generative AI, large language models, recommendation systems
   - If you must mention these concepts, be specific (e.g., vision-text alignment, hierarchical retrieval architecture, personalized generation)
   - Dig into unique angles of this week's papers that contrast sharply with previous weeks

3. **Title Styles** (choose the most appropriate):
   - Technical Breakthrough: Emphasize specific new technologies/methods (e.g., "Mamba Architecture Reshapes Sequence Modeling")
   - Application Innovation: Highlight new scenarios/domains (e.g., "From Code to Law: New Paradigm in Vertical Search")
   - Problem-Solving: Focus on key challenges solved (e.g., "Conquering Long Document Understanding Bottlenecks")
   - Trend Insight: Reveal research direction shifts (e.g., "From Scale Race to Efficiency First")

4. **Title Requirements**:
   - Format: Week {week_num}: [Core Topic]
   - Length: 15-30 words for core topic
   - Must include at least 1 specific technical term or application scenario
   - Should have "clickbait" appeal while maintaining professionalism
   - Readers should immediately see what makes this week unique

Please return in strict JSON format:
{{
    "title": "Week {week_num}: [Core Topic]",
    "reasoning": "Brief explanation of why this title was chosen (1-2 sentences)"
}}

Examples (for style reference only, do not copy):
- "Week X: Sparse Attention Mechanisms Lead Era of Million-Token Processing"
- "Week X: Zero-Shot Cross-Modal Retrieval Breaks Language Barriers"
- "Week X: Paradigm Shift from Prompt Engineering to Architecture Innovation"

/no_think
"""
}

overview_prompt = {
    "zh":"""
你是一位专业的信息检索领域专家，请基于以下一周内的论文数据，生成一段周报概览。

论文数据：
{papers_text}

周报信息：
- 周数：第{week_num}周
- 日期范围：{date_range}
- 论文数量：{total_papers}篇

请生成一段概览，要求：
1. 以"AI本周摘要:"开头
2. 总结本周信息检索领域的主要研究热点和趋势
3. 突出重要的技术突破或研究方向
4. 长度控制在150-250字
5. 语言简洁专业，符合学术周报风格
6. 避免过于技术化的细节，保持可读性

请严格按照以下JSON格式返回：
{{
    "overview": "AI本周摘要: [概览内容]"
}}

注意：
- 确保返回有效的JSON格式
- 概览应该全面概括本周的研究动态
- 突出最重要的趋势和发现
/no_think
""",
    "en":"""
You are a professional expert in the information retrieval field. Please generate a weekly report overview based on the following week's paper data.

Paper Data:
{papers_text}

Weekly Report Information:
- Week: Week {week_num}
- Date Range: {date_range}
- Number of Papers: {total_papers}

Please generate an overview with the following requirements:
1. Start with "AI Weekly Summary:"
2. Summarize the main research hotspots and trends in the information retrieval field this week
3. Highlight important technical breakthroughs or research directions
4. Length should be 150-250 words
5. Language should be concise and professional, suitable for academic weekly reports
6. Avoid overly technical details, maintain readability

Please return in the following JSON format:
{{
    "overview": "AI Weekly Summary: [Overview content]"
}}

Note:
- Ensure valid JSON format
- Overview should comprehensively summarize this week's research dynamics
- Highlight the most important trends and findings
/no_think
"""
}

highlight_prompt = {
    "zh":"""
你是一位专业的信息检索领域专家，请基于以下论文的标题和摘要，凝练生成一句话亮点。

论文标题：
{title}

论文摘要：
{abstract}

请生成一句话亮点，要求：
1. 结合标题和摘要的核心内容
2. 凝练简短，一句话即可（不超过50字）
3. 突出论文的核心贡献、创新点或重要发现
4. 语言简洁专业，具有吸引力
5. 避免直接复制摘要内容，应该是对论文亮点的提炼

请严格按照以下JSON格式返回：
{{
    "highlight": "论文亮点的一句话描述"
}}

注意：
- 确保返回有效的JSON格式
- 亮点应该凝练简短，突出论文核心价值
/no_think
""",
    "en":"""
You are a professional expert in the information retrieval field. Please generate a one-sentence highlight based on the following paper's title and abstract.

Paper Title:
{title}

Paper Abstract:
{abstract}

Please generate a one-sentence highlight with the following requirements:
1. Combine the core content of the title and abstract
2. Be concise and brief, one sentence only (no more than 50 words)
3. Highlight the paper's core contribution, innovation, or important findings
4. Language should be concise and professional, attractive
5. Avoid directly copying abstract content, should be a refinement of the paper's highlights

Please return in the following JSON format:
{{
    "highlight": "One-sentence description of the paper's highlight"
}}

Note:
- Ensure valid JSON format
- Highlight should be concise and brief, highlighting the paper's core value
/no_think
"""
}

category_summary_prompt = {
    "zh":"""
你是一位专业的信息检索领域专家，请基于以下一周内的论文分类统计数据，生成一段关于分类分布的总结。

分类统计数据：
{category_stats}

请生成一段分类总结，要求：
1. 简洁明了，1-2句话即可（不超过100字）
2. 基于实际的分类统计数据
3. 突出主要的分类分布特点或趋势
4. 语言简洁专业，符合学术周报风格

请严格按照以下JSON格式返回：
{{
    "summary": "关于本周论文分类分布的总结"
}}

注意：
- 确保返回有效的JSON格式
- 总结应该简洁有力，突出分类分布的主要特点
/no_think
""",
    "en":"""
You are a professional expert in the information retrieval field. Please generate a summary about category distribution based on the following week's paper category statistics.

Category Statistics:
{category_stats}

Please generate a category summary with the following requirements:
1. Be concise and clear, 1-2 sentences only (no more than 100 words)
2. Be based on actual category statistics
3. Highlight the main characteristics or trends of category distribution
4. Language should be concise and professional, suitable for academic weekly reports

Please return in the following JSON format:
{{
    "summary": "Summary about this week's paper category distribution"
}}

Note:
- Ensure valid JSON format
- Summary should be concise and powerful, highlighting the main characteristics of category distribution
/no_think
"""
}