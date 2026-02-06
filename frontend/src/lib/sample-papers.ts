export interface SamplePaper {
  id: string
  title: string
  authors: string
  venue: string
  year: number
  insightLevel: "Low" | "Medium" | "High" | "Very High"
  overallScore: number
  description: string
  sentences: string[]
  preAnalysis?: {
    sentences: Array<{
      id: number
      text: string
      type: "context" | "citation" | "viewpoint"
      insightScore: number | null
      isHighInsight: boolean
      citationRef: string | null
    }>
    overallAnalysis: {
      synthesisScore: number
      criticalDistance: number
      abstractionLevel: number
      overallInsightScore: number
      insightLevel: string
      summary: string
    }
  }
}

export const samplePapers: SamplePaper[] = [
  {
    id: "hiermem-transformer",
    title: "HierMem-Transformer: Hierarchical Memory for Long-Context Reasoning",
    authors: "Chen et al.",
    venue: "NeurIPS 2025",
    year: 2025,
    insightLevel: "High",
    overallScore: 4.2,
    description: "Proposes a novel architecture combining sparse attention with hierarchical memory for efficient long-context processing.",
    sentences: [
      "Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation tasks.",
      "Recent work by Brown et al. [1] introduced GPT-3, a 175B parameter model that achieves strong few-shot performance across various benchmarks.",
      "However, despite these impressive results, the quadratic complexity of self-attention mechanisms fundamentally limits the practical deployment of such models for long-context reasoning tasks.",
      "Transformer architectures [2] rely on self-attention to capture dependencies between tokens in a sequence.",
      "We argue that by combining sparse attention patterns with hierarchical memory structures, it becomes possible to achieve sub-quadratic complexity while preserving the model's ability to reason over extended contexts.",
      "This observation motivates our proposed architecture, which we term HierMem-Transformer.",
    ],
    preAnalysis: {
      sentences: [
        { id: 1, text: "Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation tasks.", type: "context", insightScore: null, isHighInsight: false, citationRef: null },
        { id: 2, text: "Recent work by Brown et al. [1] introduced GPT-3, a 175B parameter model that achieves strong few-shot performance across various benchmarks.", type: "citation", insightScore: 2, isHighInsight: false, citationRef: "[1]" },
        { id: 3, text: "However, despite these impressive results, the quadratic complexity of self-attention mechanisms fundamentally limits the practical deployment of such models for long-context reasoning tasks.", type: "viewpoint", insightScore: 4.5, isHighInsight: true, citationRef: null },
        { id: 4, text: "Transformer architectures [2] rely on self-attention to capture dependencies between tokens in a sequence.", type: "citation", insightScore: 1.5, isHighInsight: false, citationRef: "[2]" },
        { id: 5, text: "We argue that by combining sparse attention patterns with hierarchical memory structures, it becomes possible to achieve sub-quadratic complexity while preserving the model's ability to reason over extended contexts.", type: "viewpoint", insightScore: 5, isHighInsight: true, citationRef: null },
        { id: 6, text: "This observation motivates our proposed architecture, which we term HierMem-Transformer.", type: "context", insightScore: null, isHighInsight: false, citationRef: null },
      ],
      overallAnalysis: {
        synthesisScore: 3.8,
        criticalDistance: 4.5,
        abstractionLevel: 4.2,
        overallInsightScore: 4.2,
        insightLevel: "High",
        summary: "This introduction demonstrates strong argumentative depth by identifying a fundamental limitation (quadratic complexity) in existing work and proposing a novel synthesis of sparse attention and hierarchical memory. The author effectively goes beyond describing prior work to articulate why existing capabilities are insufficient.",
      },
    },
  },
  {
    id: "rag-survey",
    title: "A Survey of Retrieval-Augmented Generation: Current Methods and Future Directions",
    authors: "Wang et al.",
    venue: "ACL 2025",
    year: 2025,
    insightLevel: "Medium",
    overallScore: 3.1,
    description: "A comprehensive survey covering RAG methods with taxonomy and analysis.",
    sentences: [
      "Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for enhancing large language model outputs with external knowledge.",
      "Lewis et al. [1] first introduced the RAG framework, combining a retriever with a generator.",
      "Subsequent work [2, 3] has explored various retriever architectures including dense passage retrieval.",
      "Several studies [4, 5, 6] have investigated different fusion strategies for combining retrieved documents with model inputs.",
      "In this survey, we provide a comprehensive taxonomy of RAG methods and identify key challenges for future research.",
      "Our analysis reveals that current RAG systems struggle with multi-hop reasoning and temporal knowledge updates.",
    ],
    preAnalysis: {
      sentences: [
        { id: 1, text: "Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for enhancing large language model outputs with external knowledge.", type: "context", insightScore: null, isHighInsight: false, citationRef: null },
        { id: 2, text: "Lewis et al. [1] first introduced the RAG framework, combining a retriever with a generator.", type: "citation", insightScore: 1.5, isHighInsight: false, citationRef: "[1]" },
        { id: 3, text: "Subsequent work [2, 3] has explored various retriever architectures including dense passage retrieval.", type: "citation", insightScore: 1.8, isHighInsight: false, citationRef: "[2, 3]" },
        { id: 4, text: "Several studies [4, 5, 6] have investigated different fusion strategies for combining retrieved documents with model inputs.", type: "citation", insightScore: 2, isHighInsight: false, citationRef: "[4, 5, 6]" },
        { id: 5, text: "In this survey, we provide a comprehensive taxonomy of RAG methods and identify key challenges for future research.", type: "viewpoint", insightScore: 3.2, isHighInsight: false, citationRef: null },
        { id: 6, text: "Our analysis reveals that current RAG systems struggle with multi-hop reasoning and temporal knowledge updates.", type: "viewpoint", insightScore: 4.0, isHighInsight: true, citationRef: null },
      ],
      overallAnalysis: {
        synthesisScore: 3.5,
        criticalDistance: 2.8,
        abstractionLevel: 3.0,
        overallInsightScore: 3.1,
        insightLevel: "Medium",
        summary: "This survey introduction provides good coverage of existing work but relies heavily on listing citations without deep synthesis. The final sentence shows promising critical insight by identifying specific limitations, but earlier sentences lack comparative analysis.",
      },
    },
  },
  {
    id: "federated-learning",
    title: "FedScale: Benchmarking Model Quality in Federated Learning",
    authors: "Li et al.",
    venue: "ICML 2024",
    year: 2024,
    insightLevel: "Very High",
    overallScore: 4.7,
    description: "Introduces a comprehensive benchmarking framework that reveals fundamental trade-offs in federated learning.",
    sentences: [
      "Federated learning enables collaborative model training across decentralized data sources while preserving privacy.",
      "The seminal work by McMahan et al. [1] introduced FedAvg, which has become the de facto baseline for federated optimization.",
      "While numerous variants [2-5] have been proposed claiming superior convergence, we observe that these claims are often made under incomparable experimental conditions.",
      "More critically, existing benchmarks [6, 7] focus narrowly on convergence speed while neglecting the fundamental tension between model personalization and generalization in heterogeneous settings.",
      "We argue that this methodological fragmentation has led to a false sense of progress: improvements on one metric often come at the cost of degradation on others that remain unreported.",
      "To address this gap, we introduce FedScale, a comprehensive benchmarking framework that explicitly models the multi-objective nature of federated learning evaluation.",
      "Our framework reveals that no existing method Pareto-dominates others across all realistic deployment scenarios, fundamentally challenging the narrative of continuous improvement in the field.",
    ],
    preAnalysis: {
      sentences: [
        { id: 1, text: "Federated learning enables collaborative model training across decentralized data sources while preserving privacy.", type: "context", insightScore: null, isHighInsight: false, citationRef: null },
        { id: 2, text: "The seminal work by McMahan et al. [1] introduced FedAvg, which has become the de facto baseline for federated optimization.", type: "citation", insightScore: 2.0, isHighInsight: false, citationRef: "[1]" },
        { id: 3, text: "While numerous variants [2-5] have been proposed claiming superior convergence, we observe that these claims are often made under incomparable experimental conditions.", type: "viewpoint", insightScore: 4.8, isHighInsight: true, citationRef: "[2-5]" },
        { id: 4, text: "More critically, existing benchmarks [6, 7] focus narrowly on convergence speed while neglecting the fundamental tension between model personalization and generalization in heterogeneous settings.", type: "viewpoint", insightScore: 5.0, isHighInsight: true, citationRef: "[6, 7]" },
        { id: 5, text: "We argue that this methodological fragmentation has led to a false sense of progress: improvements on one metric often come at the cost of degradation on others that remain unreported.", type: "viewpoint", insightScore: 5.0, isHighInsight: true, citationRef: null },
        { id: 6, text: "To address this gap, we introduce FedScale, a comprehensive benchmarking framework that explicitly models the multi-objective nature of federated learning evaluation.", type: "viewpoint", insightScore: 4.2, isHighInsight: true, citationRef: null },
        { id: 7, text: "Our framework reveals that no existing method Pareto-dominates others across all realistic deployment scenarios, fundamentally challenging the narrative of continuous improvement in the field.", type: "viewpoint", insightScore: 5.0, isHighInsight: true, citationRef: null },
      ],
      overallAnalysis: {
        synthesisScore: 4.8,
        criticalDistance: 5.0,
        abstractionLevel: 4.5,
        overallInsightScore: 4.7,
        insightLevel: "Very High",
        summary: "This introduction exemplifies exceptional argumentative depth. The author identifies a meta-level problem (methodological fragmentation leading to false progress) that transcends individual technical contributions. The argument builds systematically from observation to critique to solution, with each viewpoint adding meaningful insight beyond cited work.",
      },
    },
  },
  {
    id: "text-classification",
    title: "BERT-Based Text Classification for Customer Reviews",
    authors: "Smith et al.",
    venue: "Workshop Paper 2024",
    year: 2024,
    insightLevel: "Low",
    overallScore: 1.8,
    description: "Applies BERT to customer review classification task.",
    sentences: [
      "Text classification is an important task in natural language processing.",
      "BERT [1] is a pre-trained language model that has achieved state-of-the-art results on many NLP benchmarks.",
      "Devlin et al. [1] proposed BERT which uses bidirectional training of Transformer.",
      "Many researchers [2, 3, 4] have applied BERT to various text classification tasks.",
      "In this paper, we apply BERT to classify customer reviews into positive and negative categories.",
      "We fine-tune BERT on our dataset and report the results.",
    ],
    preAnalysis: {
      sentences: [
        { id: 1, text: "Text classification is an important task in natural language processing.", type: "context", insightScore: null, isHighInsight: false, citationRef: null },
        { id: 2, text: "BERT [1] is a pre-trained language model that has achieved state-of-the-art results on many NLP benchmarks.", type: "citation", insightScore: 1.2, isHighInsight: false, citationRef: "[1]" },
        { id: 3, text: "Devlin et al. [1] proposed BERT which uses bidirectional training of Transformer.", type: "citation", insightScore: 1.0, isHighInsight: false, citationRef: "[1]" },
        { id: 4, text: "Many researchers [2, 3, 4] have applied BERT to various text classification tasks.", type: "citation", insightScore: 1.5, isHighInsight: false, citationRef: "[2, 3, 4]" },
        { id: 5, text: "In this paper, we apply BERT to classify customer reviews into positive and negative categories.", type: "viewpoint", insightScore: 2.0, isHighInsight: false, citationRef: null },
        { id: 6, text: "We fine-tune BERT on our dataset and report the results.", type: "viewpoint", insightScore: 1.5, isHighInsight: false, citationRef: null },
      ],
      overallAnalysis: {
        synthesisScore: 1.5,
        criticalDistance: 1.2,
        abstractionLevel: 2.0,
        overallInsightScore: 1.8,
        insightLevel: "Low",
        summary: "This introduction lacks argumentative depth. It merely restates existing work without identifying gaps, limitations, or opportunities for improvement. The contribution statement is purely descriptive without explaining why this application matters or what challenges it addresses.",
      },
    },
  },
  {
    id: "multimodal-reasoning",
    title: "Beyond Visual Question Answering: Towards Compositional Multimodal Reasoning",
    authors: "Zhang et al.",
    venue: "CVPR 2025",
    year: 2025,
    insightLevel: "High",
    overallScore: 4.4,
    description: "Proposes a new framework for compositional reasoning that addresses limitations of existing VQA approaches.",
    sentences: [
      "Vision-language models have achieved impressive performance on visual question answering benchmarks.",
      "Models such as CLIP [1] and BLIP [2] learn aligned visual-textual representations through contrastive learning.",
      "However, recent analysis [3] reveals that these models often exploit spurious correlations rather than performing genuine compositional reasoning.",
      "We identify three systematic failure modes: attribute binding errors, spatial relationship confusion, and temporal reasoning collapse.",
      "These failures persist even in state-of-the-art models [4, 5], suggesting that scaling alone cannot resolve fundamental architectural limitations.",
      "Drawing inspiration from cognitive science theories of compositional thought [6], we propose NeuroSymbolic Composer, a framework that explicitly decomposes complex queries into primitive reasoning operations.",
      "Our approach achieves compositional generalization to novel attribute-object combinations unseen during training, a capability that eludes purely neural approaches.",
    ],
    preAnalysis: {
      sentences: [
        { id: 1, text: "Vision-language models have achieved impressive performance on visual question answering benchmarks.", type: "context", insightScore: null, isHighInsight: false, citationRef: null },
        { id: 2, text: "Models such as CLIP [1] and BLIP [2] learn aligned visual-textual representations through contrastive learning.", type: "citation", insightScore: 1.8, isHighInsight: false, citationRef: "[1], [2]" },
        { id: 3, text: "However, recent analysis [3] reveals that these models often exploit spurious correlations rather than performing genuine compositional reasoning.", type: "viewpoint", insightScore: 4.5, isHighInsight: true, citationRef: "[3]" },
        { id: 4, text: "We identify three systematic failure modes: attribute binding errors, spatial relationship confusion, and temporal reasoning collapse.", type: "viewpoint", insightScore: 4.8, isHighInsight: true, citationRef: null },
        { id: 5, text: "These failures persist even in state-of-the-art models [4, 5], suggesting that scaling alone cannot resolve fundamental architectural limitations.", type: "viewpoint", insightScore: 4.7, isHighInsight: true, citationRef: "[4, 5]" },
        { id: 6, text: "Drawing inspiration from cognitive science theories of compositional thought [6], we propose NeuroSymbolic Composer, a framework that explicitly decomposes complex queries into primitive reasoning operations.", type: "viewpoint", insightScore: 4.5, isHighInsight: true, citationRef: "[6]" },
        { id: 7, text: "Our approach achieves compositional generalization to novel attribute-object combinations unseen during training, a capability that eludes purely neural approaches.", type: "viewpoint", insightScore: 4.2, isHighInsight: true, citationRef: null },
      ],
      overallAnalysis: {
        synthesisScore: 4.2,
        criticalDistance: 4.8,
        abstractionLevel: 4.3,
        overallInsightScore: 4.4,
        insightLevel: "High",
        summary: "This introduction demonstrates strong argumentative depth through systematic identification of failure modes and a clear connection between the problem analysis and proposed solution. The cross-disciplinary synthesis from cognitive science elevates the contribution beyond incremental improvement.",
      },
    },
  },
]
