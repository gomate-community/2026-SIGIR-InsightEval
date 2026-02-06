"""Mock data for weekly reports"""
from typing import Dict
from backend.models.schemas import WeeklyReport, BilingualText, TrendingTopic, ReportPaper

import random
import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
from backend.models.schemas import PaperResponse, TrendingTopicResponse, EngagementData


class MockDataService:
    def __init__(self):
        self.papers_data = self._generate_papers_data()
        self.topics_data = self._generate_topics_data()

    def _generate_papers_data(self) -> List[dict]:
        """生成模拟论文数据"""
        categories = ["AI", "CV", "NLP", "RO", "LG", "GN", "IR"]
        titles = [
            "Tensor Logic: The Language of AI",
            "StreamingVLM: Real-Time Understanding for Infinite Video Streams",
            "Robot Learning: A Tutorial",
            "The Art of Scaling Reinforcement Learning Compute for LLMs",
            "Diffusion Transformers with Representation Autoencoders",
            "Generative AI and Firm Productivity: Field Experiments in Online Retail",
            "Neural Information Retrieval: At the End of the Early Years",
            "Dense Passage Retrieval for Open-Domain Question Answering",
            "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction",
            "Learning Dense Representations for Entity Retrieval",
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "FiD: Fusion-in-Decoder for Open-domain Question Answering",
            "REALM: Retrieval-Augmented Language Model Pre-Training",
            "DPR: Dense Passage Retrieval for Open-Domain Question Answering",
            "ANCE: Approximate Nearest Neighbor Negative Contrastive Learning",
            "Attention Is All You Need: Transformer Architecture",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "GPT-4: Improving Language Understanding by Generative Pre-Training",
            "Vision Transformer: An Image is Worth 16x16 Words",
            "CLIP: Learning Transferable Visual Models From Natural Language",
            "T5: Text-to-Text Transfer Transformer",
            "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "ELECTRA: Pre-training Text Encoders as Discriminators",
            "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
            "DALL-E 2: Hierarchical Text-Conditional Image Generation",
            "Stable Diffusion: High-Resolution Image Synthesis",
            "ChatGPT: Optimizing Language Models for Dialogue",
            "LaMDA: Language Models for Dialog Applications",
            "PaLM: Scaling Language Modeling with Pathways",
            "Flamingo: a Visual Language Model for Few-Shot Learning"
        ]

        authors_pool = [
            ["John Smith", "Alice Johnson"],
            ["Bob Wilson", "Carol Davis", "David Brown"],
            ["Emma Thompson", "Michael Chen"],
            ["Sarah Williams", "James Rodriguez"],
            ["Lisa Anderson", "Kevin Martinez"],
            ["Jennifer Taylor", "Robert Garcia"],
            ["Mary Jackson", "Christopher Lee"],
            ["Patricia White", "Daniel Harris"],
            ["Linda Clark", "Matthew Lewis"],
            ["Barbara Walker", "Anthony Hall"]
        ]

        abstracts = [
            "This paper presents a novel approach to information retrieval using advanced neural networks. Our method demonstrates significant improvements over existing baselines.",
            "We introduce a new framework for dense passage retrieval that leverages transformer architectures to achieve state-of-the-art performance on multiple benchmarks.",
            "In this work, we explore the application of large language models to question answering tasks, showing promising results across various domains.",
            "This study investigates the effectiveness of contrastive learning in improving retrieval quality for open-domain question answering systems.",
            "We propose a unified architecture that combines the benefits of sparse and dense retrieval methods, achieving superior performance on standard evaluation datasets.",
            "Our research focuses on developing efficient indexing strategies for large-scale document collections, enabling real-time retrieval with minimal computational overhead.",
            "This paper examines the role of attention mechanisms in neural information retrieval, providing insights into model interpretability and performance optimization.",
            "We present a comprehensive evaluation of recent advances in neural ranking models, highlighting key factors that contribute to their success.",
            "This work introduces a novel training paradigm for retrieval models that incorporates user feedback to improve relevance estimation.",
            "Our study explores the application of multi-modal learning to document retrieval, demonstrating the benefits of incorporating visual and textual information."
        ]

        papers = []
        for i in range(100):  # 增加到100篇论文
            paper_id = f"2024.{random.randint(1000, 9999)}.{random.randint(10000, 99999)}"
            category = random.choice(categories)
            title = random.choice(titles)
            authors = random.choice(authors_pool)
            abstract = random.choice(abstracts)

            papers.append({
                "id": i + 1,
                "category": category,
                "title": title,
                "date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%b %d"),
                "views": f"{random.randint(100, 5000)}{'k' if random.random() > 0.7 else ''}",
                "citations": random.randint(5, 200),
                "comments": random.randint(0, 50) if random.random() > 0.3 else None,
                "score": random.randint(70, 98),
                "trending": random.randint(1, 3) if random.random() > 0.7 else None,
                # 新增字段
                "authors": authors,
                "abstract": abstract,
                "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
                "doi": f"10.48550/arXiv.{paper_id}" if random.random() > 0.3 else None,
                "journal_ref": f"Conference on {category} 2024" if random.random() > 0.5 else None,
                "primary_category": f"cs.{category}",
                "all_categories": [f"cs.{category}",
                                   f"cs.{random.choice(['AI', 'LG', 'CL'])}"] if random.random() > 0.6 else [
                    f"cs.{category}"]
            })

        return sorted(papers, key=lambda x: x["score"], reverse=True)

    def _generate_topics_data(self) -> List[dict]:
        """生成模拟话题数据"""
        sources = ["AI Research Lab", "Microsoft Research", "Google Research", "Stanford NLP", "OpenAI", "DeepMind",
                   "Meta AI", "Anthropic", "Cohere", "Hugging Face"]
        categories = ["Information Retrieval", "AI Research", "Neural IR", "Dense Retrieval", "Language Models",
                      "Computer Vision", "Natural Language Processing"]

        topics_templates = [
            {
                "topic": "RAG-Fusion: A New Paradigm for Retrieval-Augmented Generation",
                "summary": "This paper introduces RAG-Fusion, a novel approach that combines multiple retrieval strategies to enhance the quality of generated responses. The method shows significant improvements in factual accuracy and reduces hallucinations in large language models...",
                "tags": ["RAG", "Retrieval-Augmented Generation", "LLM", "Fusion"]
            },
            {
                "topic": "DeepResearch: Automated Scientific Literature Discovery and Synthesis",
                "summary": "DeepResearch presents an end-to-end system for automated literature review and synthesis. Using advanced neural retrieval and generation techniques, it can identify relevant papers, extract key insights, and generate comprehensive research summaries...",
                "tags": ["DeepResearch", "Literature Review", "Neural IR", "Research Automation"]
            },
            {
                "topic": "Generative Retrieval with Differentiable Search Index (DSI)",
                "summary": "This breakthrough work proposes treating retrieval as a generation task, where document identifiers are directly generated by neural models. DSI eliminates the need for traditional inverted indices and shows promising results on various IR benchmarks...",
                "tags": ["Generative Retrieval", "DSI", "Neural IR", "Document Indexing"]
            },
            {
                "topic": "ColBERT-v2: Efficient Multi-Vector Dense Retrieval at Scale",
                "summary": "ColBERT-v2 introduces significant improvements to the original ColBERT architecture, enabling efficient dense retrieval at web scale. The new version features better compression techniques and faster inference while maintaining high retrieval quality...",
                "tags": ["ColBERT", "Dense Retrieval", "Multi-Vector", "Efficiency"]
            },
            {
                "topic": "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking",
                "summary": "SPLADE introduces a novel approach to first-stage ranking by learning sparse representations that combine lexical matching with neural expansion. This method bridges the gap between traditional sparse retrieval and modern dense approaches...",
                "tags": ["SPLADE", "Sparse Retrieval", "Lexical Matching", "Neural Expansion"]
            },
            {
                "topic": "InstructRetro: Instruction Tuning for Enhanced Retrieval Performance",
                "summary": "InstructRetro demonstrates how instruction tuning can significantly improve retrieval model performance across diverse tasks. The approach shows remarkable zero-shot transfer capabilities and better alignment with user intent...",
                "tags": ["Instruction Tuning", "Retrieval", "Zero-shot", "User Intent"]
            },
            {
                "topic": "Multi-Modal Dense Retrieval for Visual Question Answering",
                "summary": "This work explores dense retrieval techniques for multi-modal scenarios, specifically focusing on visual question answering tasks. The proposed method effectively combines visual and textual representations for improved retrieval accuracy...",
                "tags": ["Multi-Modal", "Visual QA", "Dense Retrieval", "Vision-Language"]
            },
            {
                "topic": "Federated Learning for Privacy-Preserving Information Retrieval",
                "summary": "A comprehensive study on applying federated learning principles to information retrieval systems. The research addresses privacy concerns while maintaining retrieval effectiveness across distributed data sources...",
                "tags": ["Federated Learning", "Privacy", "Distributed IR", "Data Protection"]
            },
            {
                "topic": "Real-time Personalized Search with Neural Ranking Models",
                "summary": "This paper presents a framework for real-time personalized search using advanced neural ranking models. The system adapts to user preferences dynamically while maintaining low latency requirements...",
                "tags": ["Personalization", "Real-time Search", "Neural Ranking", "User Modeling"]
            },
            {
                "topic": "Cross-lingual Information Retrieval with Multilingual Transformers",
                "summary": "An investigation into cross-lingual information retrieval using state-of-the-art multilingual transformer models. The study covers 15 languages and demonstrates significant improvements over traditional translation-based approaches...",
                "tags": ["Cross-lingual", "Multilingual", "Transformers", "Language Transfer"]
            }
        ]

        topics = []
        for i in range(50):  # 增加到50个话题
            template = random.choice(topics_templates)
            topics.append({
                "id": i + 1,
                "topic": template["topic"],
                "source": random.choice(sources),
                "category": random.choice(categories),
                "language": "English",
                "summary": template["summary"],
                "engagement": {
                    "views": f"{random.randint(1000, 20000):,}",
                    "likes": random.randint(50, 300),
                    "retweets": random.randint(20, 150)
                },
                "heatScore": random.randint(80, 98),
                "tags": template["tags"]
            })

        return sorted(topics, key=lambda x: x["heatScore"], reverse=True)

    async def get_trending_papers(
            self,
            search: Optional[str] = None,
            category: Optional[str] = None,
            time_range: Optional[str] = "7days",
            limit: int = 20,
            offset: int = 0
    ) -> List[PaperResponse]:
        """获取热门论文"""
        # 模拟异步操作
        await asyncio.sleep(0.1)

        papers = self.papers_data.copy()

        # 根据时间范围过滤（模拟）
        if time_range == "7days":
            # 模拟最近7天的论文
            papers = [p for p in papers if random.random() > 0.3]
        elif time_range == "30days":
            # 模拟最近30天的论文
            papers = [p for p in papers if random.random() > 0.1]
        # "all" 时间范围返回所有论文

        # 搜索过滤
        if search:
            papers = [p for p in papers if
                      search.lower() in p["title"].lower() or
                      search.lower() in p["abstract"].lower() or
                      any(search.lower() in author.lower() for author in p["authors"])]

        # 分类过滤
        if category and category != "all":
            category_map = {
                "ai": "AI",
                "cv": "CV",
                "nlp": "NLP",
                "ro": "RO",
                "lg": "LG",
                "gn": "GN",
                "ir": "IR"
            }
            target_category = category_map.get(category.lower(), category.upper())
            papers = [p for p in papers if p["category"] == target_category]

        # 分页
        papers = papers[offset:offset + limit]

        return [PaperResponse(**paper) for paper in papers]

    async def get_trending_topics(
            self,
            search: Optional[str] = None,
            source: Optional[str] = None,
            time_range: Optional[str] = "1week",
            sort_by: Optional[str] = "default",
            limit: int = 20,
            offset: int = 0
    ) -> List[TrendingTopicResponse]:
        """获取热门话题"""
        # 模拟异步操作
        await asyncio.sleep(0.1)

        topics = self.topics_data.copy()

        # 搜索过滤
        if search:
            topics = [t for t in topics if search.lower() in t["topic"].lower() or
                      search.lower() in t["summary"].lower()]

        # 来源过滤
        if source and source != "all":
            source_map = {
                "ir": "Information Retrieval",
                "ai": "AI Research",
                "neural": "Neural IR",
                "dense": "Dense Retrieval"
            }
            if source in source_map:
                topics = [t for t in topics if t["category"] == source_map[source]]

        # 排序
        if sort_by == "hot":
            topics = sorted(topics, key=lambda x: x["heatScore"], reverse=True)
        elif sort_by == "latest":
            # 模拟按时间排序
            random.shuffle(topics)
        elif sort_by == "engagement":
            topics = sorted(topics, key=lambda x: x["engagement"]["likes"], reverse=True)

        # 分页
        topics = topics[offset:offset + limit]

        return [TrendingTopicResponse(**topic) for topic in topics]

    async def generate_chat_response(self, question: str) -> str:
        """生成聊天响应"""
        # 模拟异步操作
        await asyncio.sleep(0.5)

        # 根据问题生成相关回答
        responses = [
            f"Based on recent research trends, your question about '{question}' is very relevant to current developments in information retrieval.",
            f"I found several papers related to your query '{question}'. The most relevant ones discuss advanced neural approaches and their applications.",
            f"Regarding '{question}', the latest research shows promising results in combining traditional IR methods with modern deep learning techniques.",
            f"Your question about '{question}' touches on an important area. Recent papers have explored this topic extensively, particularly in the context of large language models.",
            f"The topic '{question}' has been gaining attention in the IR community. Several recent publications have proposed novel approaches to address related challenges."
        ]

        base_response = random.choice(responses)

        # 添加更详细的内容
        detailed_content = """

Here are some key insights from recent papers:

1. **Neural Retrieval Methods**: Modern approaches leverage transformer architectures to improve retrieval effectiveness.

2. **Dense vs Sparse Retrieval**: There's an ongoing debate about the trade-offs between dense and sparse retrieval methods.

3. **Evaluation Metrics**: New evaluation frameworks are being developed to better assess retrieval quality.

4. **Real-world Applications**: Industry applications are driving innovation in scalable retrieval systems.

Would you like me to elaborate on any of these points or discuss specific papers related to your question?"""

        return base_response + detailed_content

    def get_weekly_reports(self) -> Dict[str, WeeklyReport]:
        """Get all weekly reports mock data"""
        return {
            "70": WeeklyReport(
                id="70",
                week="Issue 70",
                dateRange="Oct 28 - Nov 3",
                publishDate="10-31",
                totalPapers=47,
                topicsCount=4,
                summary=BilingualText(
                    zh="本周AI领域的焦点集中在AI代理的发展和应用。从OpenAI和GitHub的代理平台发布,到阿里云和高通的实践经验分享,再到理想汽车的VLA自动驾驶模型,一个更加智能和自主的AI生态系统正在形成。",
                    en="This week's focus in AI centers on the development and application of AI agents. From OpenAI and GitHub's agent platform announcements to practical experiences from Alibaba Cloud and Qualcomm, and Li Auto's VLA autonomous driving model, a more intelligent and autonomous AI ecosystem is emerging."
                ),
                highlights=[
                    BilingualText(zh="AI编程", en="AI Coding"),
                    BilingualText(zh="密集检索", en="Dense Retrieval"),
                    BilingualText(zh="RAG系统", en="RAG Systems"),
                    BilingualText(zh="神经重排序", en="Neural Re-ranking")
                ]
            ),
            "69": WeeklyReport(
                id="69",
                week="Issue 69",
                dateRange="Oct 21 - Oct 27",
                publishDate="10-24",
                totalPapers=42,
                topicsCount=5,
                summary=BilingualText(
                    zh="本周聚焦AI领域的核心洞察,围绕智能创造力、开发范式和上下文工程的讨论引起广泛关注。同时,模型研究在长文本处理和评估方法上取得新突破。",
                    en="This week focuses on key insights in the AI field, with discussions around intelligent creativity, development paradigms, and context engineering gathering significant attention. Meanwhile, model research has made new breakthroughs in long-text processing and evaluation methods."
                ),
                highlights=[
                    BilingualText(zh="长文本处理", en="Long Text Processing"),
                    BilingualText(zh="开发范式", en="Development Paradigms"),
                    BilingualText(zh="AI原生应用", en="AI-Native Apps"),
                    BilingualText(zh="产品创新", en="Product Innovation")
                ]
            ),
            "68": WeeklyReport(
                id="68",
                week="Issue 68",
                dateRange="Oct 14 - Oct 20",
                publishDate="10-17",
                totalPapers=39,
                topicsCount=3,
                summary=BilingualText(
                    zh="本周AI领域依然精彩纷呈。从模型架构的突破性创新到业务应用的实战落地,我们见证了人工智能发展加速的洞察。",
                    en="The AI field remains exciting this week. From breakthrough innovations in model architecture to real-world business applications, we witness insights into the accelerating development of artificial intelligence."
                ),
                highlights=[
                    BilingualText(zh="模型架构", en="Model Architecture"),
                    BilingualText(zh="检索优化", en="Retrieval Optimization"),
                    BilingualText(zh="工具开发", en="Tool Development"),
                    BilingualText(zh="突破性创新", en="Breakthrough Innovation")
                ]
            ),
            "67": WeeklyReport(
                id="67",
                week="Issue 67",
                dateRange="Oct 7 - Oct 13",
                publishDate="10-10",
                totalPapers=51,
                topicsCount=6,
                summary=BilingualText(
                    zh="本周聚焦OpenAI DevDay带来的创新浪潮,为整个行业注入新活力。一系列重大发布预示着AI发展的新纪元,各大公司也在模型更新上持续创新。",
                    en="This week focuses on the innovation wave brought by OpenAI DevDay, which has energized the entire industry. A series of major releases signals a new era of AI development, while major companies continue to innovate with model updates."
                ),
                highlights=[
                    BilingualText(zh="OpenAI DevDay", en="OpenAI DevDay"),
                    BilingualText(zh="多模态检索", en="Multi-modal Retrieval"),
                    BilingualText(zh="AI产品应用", en="AI Product Applications"),
                    BilingualText(zh="模型竞争", en="Model Competition")
                ]
            ),
            "66": WeeklyReport(
                id="66",
                week="Issue 66",
                dateRange="Sep 30 - Oct 6",
                publishDate="10-02",
                totalPapers=44,
                topicsCount=5,
                summary=BilingualText(
                    zh="本周AI领域持续繁荣,学术研究和工业应用都有新进展。特别是在检索系统优化和实时应用场景方面,涌现出许多创新思路。",
                    en="The AI field continues to thrive this week, with new developments across academic research and industrial applications. Particularly in retrieval system optimization and real-time application scenarios, many innovative ideas have emerged."
                ),
                highlights=[
                    BilingualText(zh="实时检索", en="Real-time Retrieval"),
                    BilingualText(zh="系统优化", en="System Optimization"),
                    BilingualText(zh="实践落地", en="Practical Implementation"),
                    BilingualText(zh="评估方法", en="Evaluation Methods")
                ]
            )
        }


    def get_report_detail(self,report_id: str) -> WeeklyReport:
        """Get report detail mock data by report ID"""
        reports = {
            "70": WeeklyReport(
                id="70",
                week="Issue 70",
                dateRange="Oct 28 - Nov 3",
                publishDate="11-03",
                totalPapers=47,
                topicsCount=4,
                summary=BilingualText(
                    zh="本周信息检索领域共有47篇论文，涵盖4个热点话题。",
                    en="This week's information retrieval field has 47 papers covering 4 hot topics."
                ),
                highlights=[
                    BilingualText(zh="密集检索", en="Dense Retrieval"),
                    BilingualText(zh="RAG系统", en="RAG Systems"),
                    BilingualText(zh="神经重排序", en="Neural Re-ranking"),
                    BilingualText(zh="多模态信息检索", en="Multi-modal IR")
                ],
                trendingTopics=[
                    TrendingTopic(
                        name=BilingualText(zh="密集检索", en="Dense Retrieval"),
                        count=12,
                        growth="+35%"
                    ),
                    TrendingTopic(
                        name=BilingualText(zh="RAG系统", en="RAG Systems"),
                        count=9,
                        growth="+50%"
                    ),
                    TrendingTopic(
                        name=BilingualText(zh="神经重排序", en="Neural Re-ranking"),
                        count=8,
                        growth="+12%"
                    ),
                    TrendingTopic(
                        name=BilingualText(zh="多模态信息检索", en="Multi-modal IR"),
                        count=6,
                        growth="+25%"
                    )
                ],
                keyInsights=[
                    BilingualText(
                        zh="检索增强生成(RAG)技术持续升温,相关论文本周增长50%,主要聚焦于提升检索质量和生成一致性",
                        en="Retrieval-Augmented Generation (RAG) technology continues to heat up with a 50% increase in related papers this week, primarily focusing on improving retrieval quality and generation consistency"
                    ),
                    BilingualText(
                        zh="密集检索仍是主流方向,但研究重点已从单纯提高召回率转向优化检索效率和跨语言应用",
                        en="Dense retrieval remains the mainstream direction, but research focus has shifted from purely improving recall to optimizing retrieval efficiency and cross-lingual applications"
                    ),
                    BilingualText(
                        zh="多模态信息检索新趋势:文本-图像-视频联合检索架构受到更多关注",
                        en="New trends in multi-modal information retrieval: text-image-video joint retrieval architectures are receiving more attention"
                    ),
                    BilingualText(
                        zh="评估指标创新:多篇论文提出针对生成式检索的新型评估方法,解决传统指标在评估RAG系统时的不足",
                        en="Evaluation metric innovation: multiple papers propose new evaluation methods for generative retrieval, addressing the shortcomings of traditional metrics in evaluating RAG systems"
                    )
                ],
                topPapers=[
                    ReportPaper(
                        id=1,
                        title=BilingualText(
                            zh="高效密集检索与学习稀疏表示",
                            en="Efficient Dense Retrieval with Learned Sparse Representations"
                        ),
                        authors="Zhang et al.",
                        institution="Stanford University",
                        arxivId="2311.12345",
                        highlight=BilingualText(
                            zh="提出混合稀疏-密集表示方法,在保持检索质量的同时将推理速度提升3倍",
                            en="Proposes a hybrid sparse-dense representation method that achieves 3x inference speedup while maintaining retrieval quality"
                        ),
                        category=BilingualText(zh="密集检索", en="Dense Retrieval"),
                        abstract=BilingualText(
                            zh="本文提出了一种新颖的信息检索方法,结合了稀疏表示和密集表示的优势。通过学习最优的混合表示,我们在检索质量和计算效率方面都取得了显著改进。",
                            en="This paper presents a novel information retrieval approach that combines the advantages of sparse and dense representations. By learning optimal hybrid representations, we achieve significant improvements in both retrieval quality and computational efficiency."
                        )
                    ),
                    ReportPaper(
                        id=2,
                        title=BilingualText(
                            zh="RAG-Eval: 检索增强生成的综合基准测试",
                            en="RAG-Eval: A Comprehensive Benchmark for Retrieval-Augmented Generation"
                        ),
                        authors="Chen et al.",
                        institution="MIT & Google Research",
                        arxivId="2311.23456",
                        highlight=BilingualText(
                            zh="构建了首个系统性的RAG评估基准,涵盖6个维度、12个数据集,填补了该领域评估标准的空白",
                            en="Builds the first systematic RAG evaluation benchmark covering 6 dimensions and 12 datasets, filling the gap in evaluation standards for this field"
                        ),
                        category=BilingualText(zh="评估", en="Evaluation"),
                        abstract=BilingualText(
                            zh="我们提出了RAG-Eval,这是一个全面的基准测试,旨在从准确性、一致性和计算效率等多个维度评估检索增强生成系统。",
                            en="We present RAG-Eval, a comprehensive benchmark designed to evaluate retrieval-augmented generation systems across multiple dimensions including accuracy, consistency, and computational efficiency."
                        )
                    ),
                    ReportPaper(
                        id=3,
                        title=BilingualText(
                            zh="统一语义空间的跨模态检索",
                            en="Cross-Modal Retrieval with Unified Semantic Space"
                        ),
                        authors="Liu et al.",
                        institution="Tsinghua University",
                        arxivId="2311.34567",
                        highlight=BilingualText(
                            zh="提出统一语义空间框架,实现文本、图像、音频的无缝跨模态检索,在多个基准测试中达到SOTA",
                            en="Proposes a unified semantic space framework enabling seamless cross-modal retrieval across text, images, and audio, achieving SOTA on multiple benchmarks"
                        ),
                        category=BilingualText(zh="多模态信息检索", en="Multi-modal IR"),
                        abstract=BilingualText(
                            zh="我们的统一语义空间框架实现了文本、图像和音频之间的无缝跨模态检索,在多个基准数据集上达到了最先进的性能。",
                            en="Our unified semantic space framework enables seamless cross-modal retrieval between text, images, and audio, achieving state-of-the-art performance on multiple benchmark datasets."
                        )
                    )
                ],
                emergingKeywords=[
                    BilingualText(zh="语义缓存", en="Semantic Cache"),
                    BilingualText(zh="查询重写", en="Query Rewriting"),
                    BilingualText(zh="混合搜索", en="Hybrid Search"),
                    BilingualText(zh="后期交互", en="Late Interaction"),
                    BilingualText(zh="对比学习", en="Contrastive Learning"),
                    BilingualText(zh="零样本检索", en="Zero-shot Retrieval")
                ]
            )
        }

        # Return default report 70 if report_id not found
        return reports.get(report_id, reports["70"])

