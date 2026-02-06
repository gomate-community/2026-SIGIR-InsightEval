CATEGORY_PROMPT = """
你是一位专业的信息检索领域专家，请根据论文的标题和摘要，将其分类到最合适的IR分类体系中。

论文信息：
标题：{title}
摘要：{abstract}

可选的分类体系如下：
{categories_str}

请仔细分析论文内容，选择最合适的主分类，并从该分类的子类别中选择1-3个最相关的子类别。

请严格按照以下JSON格式返回分类结果：
{{
    "category_name": "选择的主分类名称",
    "category_description": "主分类的描述",
    "sub_categories": ["子类别1", "子类别2", "子类别3"],
    "confidence": 分类置信度(0-100),
    "reasoning": "分类理由"
}}

注意：
- category_name必须完全匹配上述分类体系中的分类名称
- sub_categories必须从对应主分类的子类别中选择
- confidence是你对分类结果的置信度(0-100)
- reasoning简要说明为什么选择这个分类
- 如果论文不属于信息检索领域，请选择最相近的分类
- 确保返回有效的JSON格式
- 如果摘要不存在，请根据标题以及分类体系的描述作为参考进行类别判断，不要盲目判断
/no_think
"""