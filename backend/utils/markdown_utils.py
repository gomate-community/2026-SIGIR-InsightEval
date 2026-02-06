import re


def remove_markdown_split_lines(md_text: str) -> str:
    """
    去除 Markdown 文本中整行的 --- 分割线，但保留表格、代码块或正文中的 ---。

    参数:
        md_text (str): 输入的 Markdown 字符串

    返回:
        str: 去除分割线后的 Markdown 字符串
    """
    lines = md_text.splitlines()
    cleaned_lines = []
    in_code_block = False  # 标记是否在代码块内 (``` 或 ~~~)

    for line in lines:
        stripped = line.strip()

        # 检测是否进入或退出代码块
        if re.match(r"^(```|~~~)", stripped):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            continue

        # 若在代码块中，不处理
        if in_code_block:
            cleaned_lines.append(line)
            continue

        # 匹配整行都是分割线（只含 -, 可含空格）
        # 但排除表格分隔线（含有 | 或 :）
        if re.match(r"^[-\s]{3,}$", stripped) and '|' not in stripped and ':' not in stripped:
            cleaned_lines.append("")  # 用换行替代
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def fix_markdown_bold_spacing(text: str) -> str:
    """
    在 Markdown 加粗语法 **xxx** 后自动补充一个空格，以及内部不要有空格
    如果后面没有空格或标点符号（中英文均考虑）。
    一个是 要加粗内容内部不要有空格，比如**xxx**， xx与两个星号不要有空格
    一个是 保证加粗内容外部有空格，第一个**的左侧以及第一个**的右侧，比如yy **xxx** zz,**xx**要与yy和zz有空格
    """

    # 找出所有加粗内容（最小匹配）
    bold_pattern = r'\*\*(.+?)\*\*'
    parts = re.split(f'({bold_pattern})', text)

    result = []
    for i, part in enumerate(parts):
        # 检查是否是加粗内容
        match = re.match(bold_pattern, part)
        if match:
            # 处理加粗内容：去除内部空格
            content = match.group(1).strip()
            bold_text = f'**{content}**'

            # 处理左侧空格
            if i > 0 and result:
                prev_text = result[-1]
                # 如果前面有内容且最后一个字符不是空格，添加空格
                if prev_text and not prev_text.endswith(' ') and not prev_text.endswith('\n'):
                    result[-1] = prev_text + ' '

            result.append(bold_text)

            # 处理右侧空格
            if i + 1 < len(parts):
                next_part = parts[i + 1] if i + 1 < len(parts) else ''
                # 如果后面有内容且第一个字符不是空格或换行，需要添加空格
                if next_part and not next_part.startswith(' ') and not next_part.startswith('\n'):
                    result.append(' ')
        else:
            # 普通文本直接添加
            if part:  # 跳过空字符串
                result.append(part)

    return ''.join(result)
def remove_markdown_code_blocks(content):
    """
    智能去除markdown代码块包裹，但保留内容中的代码块

    Args:
        content (str): 包含markdown代码块标记的文本内容

    Returns:
        str: 清理后的文本内容
    """
    # 去除前后的空白字符
    content = content.strip()
    
    # 检查是否是被markdown代码块完整包裹的情况
    is_wrapped_by_markdown = False
    
    # 检查开头
    if content.startswith('```markdown\n') or content.startswith('```markdown '):
        # 确认这是一个完整的markdown包裹
        if content.endswith('\n```') or content.endswith('```'):
            # 进一步验证：去除开头和结尾后，内容中不应该有未配对的```
            inner_content = content[11:].rstrip('```').strip()
            
            # 简单检查：如果内容以```开头，说明可能不是完整包裹
            if not inner_content.startswith('```'):
                is_wrapped_by_markdown = True
    
    # 如果确认是被markdown包裹的，则去除包裹
    if is_wrapped_by_markdown:
        content = content[11:]  # 去除```markdown\n
        if content.endswith('\n```'):
            content = content[:-4]  # 去除\n```
        elif content.endswith('```'):
            content = content[:-3]  # 去除```
    
    # 处理其他简单的```包裹情况（非markdown）
    elif content.startswith('```') and content.endswith('```'):
        lines = content.split('\n')
        if len(lines) >= 2:
            # 检查第一行是否只是语言标识符
            first_line = lines[0].strip()
            if first_line.startswith('```') and len(first_line) <= 20:
                # 检查内容中是否还有其他代码块
                inner_content = '\n'.join(lines[1:-1]) if len(lines) > 2 else ''
                
                # 如果内容中没有其他```代码块，则可以安全去除包裹
                if '```' not in inner_content:
                    content = inner_content
    
    return content.strip()


# if __name__ == '__main__':


#     content="""
#     ```markdown
#     ### 1.2.3 交互式检索与多语言支持（2022-2023年）
#
# 随着用户交互和多语言需求的增长，RAG技术开始向更灵活、更智能的方向发展。这一阶段的代表性工作包括论文2中提出的QueryExplorer，它构建了一个交互式查询生成与重构工具，支持用户反馈和多语言检索。该工具结合HITL（Human-in-the-Loop）机制，提升了检索系统的适应性和用户体验。
#
# 此外，多语言RAG系统的研究也逐渐增多，尤其是在跨语言信息检索和跨语言问答任务中，RAG技术展现出强大的潜力。这一阶段的技术方法强调检索与生成的协同优化，以及用户行为建模，为后续的动态场景建模和任务规划提供了基础。
#
# ### 1.2.4 动态场景建模与多模态融合（2023年至今）
#
# 近年来，RAG技术的应用场景不断扩展，尤其是在动态场景建模、虚拟现实、增强现实和自动驾驶等领域。论文3中综述了动态场景表示与重建的最新进展，重点探讨了NeRF（Neural Radiance Fields）和3DGS（3D Gaussian Splatting）等技术在动态场景中的应用。这些方法通过引入变形场、流场和4D表示，实现了对动态对象的高精度建模。
#
# 同时，多模态RAG系统也逐渐成为研究热点。例如，论文4探讨了查询生成策略对新闻真实性验证的影响，结合用户行为分析和模拟实验，提出了优化搜索行为的方法。这一阶段的技术方法强调生成模型与检索系统的深度融合，以及对用户意图和任务目标的动态响应。
#
# ### 1.2.5 未来方向与挑战
#
# 当前，RAG技术正处于快速发展阶段，但仍面临诸多挑战。例如，如何在动态环境中实现高效、准确的检索与生成，如何提升多语言和多模态系统的鲁棒性，以及如何通过弱监督学习和自监督学习降低标注成本。此外，随着生成模型能力的提升，如何确保生成内容的可信度和可解释性，也成为研究的重要方向。
#
# ---
#
# ```mermaid
# timeline
#     title RAG技术演变时间线
#     dateFormat  YYYY
#     section 早期探索与基础框架
#         基于规则与静态知识库的系统 :2010
#     section 深度学习与神经符号方法的融合
#         DPR, REALM, CECI模型 :2019
#     section 交互式检索与多语言支持
#         QueryExplorer, 多语言RAG系统 :2022
#     section 动态场景建模与多模态融合
#         NeRF, 3DGS, 多模态RAG系统 :2023
#     section 未来方向与挑战
#         多语言、弱监督、可信生成 :2024
# ```
# ```
#     """
#
#     content=remove_markdown_code_blocks(content)
#     print(content)
