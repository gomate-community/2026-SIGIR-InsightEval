/**
 * 前端 API 服务模块
 * 封装对后端 Insight Analysis API 的所有调用
 */

// API 基础 URL，开发环境使用 Vite 代理
const API_BASE_URL = '/api/insight';

// ============== 类型定义 ==============

export interface InsightScores {
    synthesis: number;
    critical: number;
    abstraction: number;
}

export interface Evidence {
    quote: string;
    source: string;
    criteria: string;
}

export interface AnalyzedSentence {
    id: number;
    text: string;
    type: 'context' | 'citation' | 'viewpoint';
    insightLevel: 'low' | 'medium' | 'high';
    scores: InsightScores;
    analysis: string;
    source?: string;
    evidence?: Evidence[];
}

export interface InsightReport {
    summary: string;
    strengths: string[];
    weaknesses: string[];
    overall_score: number;
}

export interface AnalysisResponse {
    sentences: AnalyzedSentence[];
    overallScore: number;
    summary: string;
    paperTitle?: string;
    report?: InsightReport;
}

export interface AnalysisProgress {
    status: 'pending' | 'processing' | 'completed' | 'error';
    progress: number;
    message: string;
    taskId?: string;
}

// ============== API 函数 ==============

/**
 * 健康检查
 */
export async function healthCheck(): Promise<{ status: string; service: string; version: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
}

/**
 * 上传 PDF 并执行完整分析
 * @param file PDF 文件
 * @param onProgress 可选的进度回调
 */
export async function analyzePdf(
    file: File,
    onProgress?: (step: string, percent: number) => void
): Promise<AnalysisResponse> {
    onProgress?.('正在上传文件...', 10);

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/analyze/pdf`, {
        method: 'POST',
        body: formData,
    });

    onProgress?.('正在分析论文...', 50);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Analysis failed: ${response.status}`);
    }

    onProgress?.('分析完成', 100);
    return response.json();
}

/**
 * 分析文本（Introduction 部分）
 * @param text 要分析的文本
 * @param paperTitle 可选的论文标题
 */
export async function analyzeText(
    text: string,
    paperTitle?: string
): Promise<AnalysisResponse> {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text,
            paperTitle,
        }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Analysis failed: ${response.status}`);
    }

    return response.json();
}

/**
 * 异步分析（用于长文本）
 */
export async function analyzeTextAsync(
    text: string,
    paperTitle?: string
): Promise<{ taskId: string; status: string }> {
    const response = await fetch(`${API_BASE_URL}/analyze/async`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text,
            paperTitle,
        }),
    });

    if (!response.ok) {
        throw new Error(`Async analysis failed: ${response.status}`);
    }

    return response.json();
}

/**
 * 获取异步任务状态
 */
export async function getAnalysisStatus(taskId: string): Promise<AnalysisProgress> {
    const response = await fetch(`${API_BASE_URL}/analyze/status/${taskId}`);
    if (!response.ok) {
        throw new Error(`Failed to get status: ${response.status}`);
    }
    return response.json();
}

/**
 * 获取异步分析结果
 */
export async function getAnalysisResult(taskId: string): Promise<AnalysisResponse> {
    const response = await fetch(`${API_BASE_URL}/analyze/result/${taskId}`);
    if (!response.ok) {
        throw new Error(`Failed to get result: ${response.status}`);
    }
    return response.json();
}

// ============== 辅助函数 ==============

/**
 * 计算平均洞察力分数
 */
export function calculateAverageScore(scores: InsightScores): number {
    return (scores.synthesis + scores.critical + scores.abstraction) / 3;
}

/**
 * 判断是否为高洞察力
 */
export function isHighInsight(sentence: AnalyzedSentence): boolean {
    return sentence.insightLevel === 'high';
}

/**
 * 将后端响应转换为前端展示需要的格式
 */
export function transformForDisplay(response: AnalysisResponse) {
    return {
        sentences: response.sentences.map((s) => ({
            ...s,
            insightScore: calculateAverageScore(s.scores),
            isHighInsight: isHighInsight(s),
            citationRef: s.source || null,
        })),
        overallAnalysis: {
            synthesisScore: response.sentences.length > 0
                ? response.sentences.reduce((acc, s) => acc + s.scores.synthesis, 0) / response.sentences.length
                : 0,
            criticalDistance: response.sentences.length > 0
                ? response.sentences.reduce((acc, s) => acc + s.scores.critical, 0) / response.sentences.length
                : 0,
            abstractionLevel: response.sentences.length > 0
                ? response.sentences.reduce((acc, s) => acc + s.scores.abstraction, 0) / response.sentences.length
                : 0,
            overallInsightScore: response.overallScore,
            insightLevel: response.report?.overall_score
                ? (response.report.overall_score >= 7 ? 'High' : response.report.overall_score >= 4 ? 'Medium' : 'Low')
                : 'Medium',
            summary: response.summary,
            report: response.report,
        },
        paperTitle: response.paperTitle,
    };
}
