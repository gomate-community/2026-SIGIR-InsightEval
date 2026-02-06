AFFILIATION_EXTRACTION_PROMPT = """
你是一位专业的学术论文分析助手。请从提供的论文文本中提取作者姓名、机构信息以及相关邮箱。

返回一个JSON对象列表。每个对象必须包含以下字段：
* 'author': 作者姓名。
* 'author_email': 该作者的邮箱地址，如果文中未明确提及则为空字符串。
* 'org': 该作者对应的具体机构/单位全称，不要猜测未明确提及的信息。
* 'org_email': 该机构的联系邮箱，如果文中未明确提及则为空字符串。
* 'is_industry': 布尔值，true表示机构为企业/公司（如Google、Meta、Microsoft、DeepMind等），false表示机构为大学或政府研究机构。

请确保根据文本中的上标、格式等信息正确对齐作者与机构关系。

论文文本：
{text}

**Example：**
```json
[
  {{
    "author": "Kaiming He",
    "author_email": "kaiming@fb.com",
    "org": "Facebook AI Research (FAIR)",
    "org_email": "contact@fair.fb.com",
    "is_industry": true
  }},
  {{
    "author": "Ross Girshick",
    "author_email": "ross@fb.com",
    "org": "Facebook AI Research (FAIR)",
    "org_email": "contact@fair.fb.com",
    "is_industry": true
  }},
  {{
    "author": "Sample Student",
    "author_email": "student@tsinghua.edu.cn",
    "org": "Tsinghua University",
    "org_email": "",
    "is_industry": false
  }}
]
```

请仅返回JSON数组，不要添加其他文本或解释。
/no_think

"""