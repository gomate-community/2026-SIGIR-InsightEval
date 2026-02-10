"""
Local Insight Analysis Prompts - 本地引用论文洞察力分析提示词

复用 insight_prompt.py 中的通用提示词，新增本地引用证据提取提示词
"""

# 从原有 prompt 中复用
from backend.prompts.insight_prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    VIEWPOINT_EXTRACTION_PROMPT,
    SCORING_WITH_EVIDENCE_PROMPT,
    GLOBAL_REPORT_SYSTEM_PROMPT,
)

# 新增: 从本地引用论文中提取证据的提示词
EVIDENCE_EXTRACTION_PROMPT = """You are an expert at analyzing academic papers and extracting supporting evidence.

Given a **Viewpoint Sentence** from a paper under review, and the **Full Text** of a referenced paper, extract the most relevant supporting evidence from the reference paper.

## Viewpoint Sentence
"{viewpoint}"

## Reference Paper: {ref_name}
{ref_text}

## Task
1. Find passages in the reference paper that directly support, contradict, or relate to the viewpoint sentence.
2. Extract 1-3 most relevant quotes (each quote should be 1-3 sentences).
3. For each quote, explain how it relates to the viewpoint (supports, contradicts, provides context, etc.).

## Output Format
Return a JSON array:
```json
[
  {{
    "quote": "exact quote from the reference paper",
    "relevance": "supports|contradicts|provides_context",
    "explanation": "brief explanation of how this evidence relates to the viewpoint"
  }}
]
```

If no relevant evidence is found, return an empty array: []

IMPORTANT: Return ONLY valid JSON."""


# 新增: 观点句筛选提示词（分析全文句子，筛选出观点句）
VIEWPOINT_FILTER_PROMPT = """You are an expert academic paper analyst. Analyze the following sentences extracted from a paper's Introduction section.

## Paper Title: {paper_title}

## Sentences to Analyze
{sentences_json}

## Task
For each sentence, determine:
1. **Type**: Is it `context` (background), `citation` (citing others' work), or `viewpoint` (author's own insight/argument)?
2. **Citation Numbers**: Extract any citation numbers like [1], [2-5], etc.
3. **Brief Analysis**: Why you classified it this way.

## Scoring (for viewpoint sentences only)
Rate on 3 dimensions (1.0-5.0):
- **synthesis**: Does it connect multiple ideas?
- **critical**: Does it show critical thinking?
- **abstraction**: Does it generalize beyond specifics?

## Output Format
Return a JSON array:
```json
[
  {{
    "id": 1,
    "text": "sentence text",
    "type": "context|citation|viewpoint",
    "citation_numbers": [1, 2],
    "scores": {{"synthesis": 2.5, "critical": 3.0, "abstraction": 2.0}},
    "analysis": "brief explanation"
  }}
]
```

IMPORTANT: Return ONLY valid JSON."""
