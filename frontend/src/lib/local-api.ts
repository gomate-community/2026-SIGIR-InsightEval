/**
 * Local Insight Analysis SSE API Client
 * 
 * 通过 Server-Sent Events 与后端 /api/local-insight/analyze/stream 交互
 * 实时接收 4 个步骤的分析结果
 */

const API_BASE_URL = '/api/local-insight';

// ============== 类型定义 ==============

export interface Step1Data {
    total_sentences: number;
    cited_sentences: number;
    sentences: Array<{
        text: string;
        has_citation: boolean;
        citation_numbers: number[];
    }>;
    introduction: string;
}

export interface ViewpointEvidence {
    quote: string;
    source: string;
    relevance?: string;
    explanation?: string;
}

export interface ViewpointWithEvidence {
    id: number;
    text: string;
    citation_numbers: number[];
    evidence: ViewpointEvidence[];
    analysis: string;
}

export interface Step2Data {
    total_viewpoints: number;
    viewpoints: ViewpointWithEvidence[];
}

export interface ScoredViewpoint {
    id: number;
    text: string;
    scores: { synthesis: number; critical: number; abstraction: number };
    analysis: string;
    insight_level: string;
    evidence: ViewpointEvidence[];
}

export interface Step3Data {
    scored_viewpoints: ScoredViewpoint[];
    avg_score: number;
}

export interface Step4Data {
    summary: string;
    strengths: string[];
    weaknesses: string[];
    overall_score: number;
}

export interface ProgressData {
    step: number;
    message: string;
}

export type SSEEventType = 'progress' | 'step1' | 'step2' | 'step3' | 'step4' | 'done' | 'error';

export interface SSECallbacks {
    onProgress?: (data: ProgressData) => void;
    onStep1?: (data: Step1Data) => void;
    onStep2?: (data: Step2Data) => void;
    onStep3?: (data: Step3Data) => void;
    onStep4?: (data: Step4Data) => void;
    onDone?: (data: { message: string; overall_score: number }) => void;
    onError?: (data: { message: string }) => void;
}

// ============== SSE Client ==============

/**
 * 发起流式分析请求
 * 
 * @param file 待评估论文 PDF
 * @param referencesDir 引用论文文件夹路径
 * @param callbacks SSE 事件回调
 * @returns AbortController（用于取消请求）
 */
export function analyzeLocalPaper(
    file: File,
    referencesDir: string,
    callbacks: SSECallbacks,
): AbortController {
    const controller = new AbortController();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('references_dir', referencesDir);

    fetch(`${API_BASE_URL}/analyze/stream`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
    })
        .then(async (response) => {
            if (!response.ok) {
                const errBody = await response.json().catch(() => ({}));
                callbacks.onError?.({ message: errBody.detail || `HTTP ${response.status}` });
                return;
            }

            const reader = response.body?.getReader();
            if (!reader) {
                callbacks.onError?.({ message: 'ReadableStream not supported' });
                return;
            }

            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Parse SSE: lines separated by \n\n
                const parts = buffer.split('\n\n');
                buffer = parts.pop() || '';

                for (const part of parts) {
                    const lines = part.split('\n');
                    let eventType = 'message';
                    let eventData = '';

                    for (const line of lines) {
                        if (line.startsWith('event: ')) {
                            eventType = line.slice(7).trim();
                        } else if (line.startsWith('data: ')) {
                            eventData = line.slice(6);
                        }
                    }

                    if (!eventData) continue;

                    try {
                        const parsed = JSON.parse(eventData);
                        switch (eventType) {
                            case 'progress':
                                callbacks.onProgress?.(parsed);
                                break;
                            case 'step1':
                                callbacks.onStep1?.(parsed);
                                break;
                            case 'step2':
                                callbacks.onStep2?.(parsed);
                                break;
                            case 'step3':
                                callbacks.onStep3?.(parsed);
                                break;
                            case 'step4':
                                callbacks.onStep4?.(parsed);
                                break;
                            case 'done':
                                callbacks.onDone?.(parsed);
                                break;
                            case 'error':
                                callbacks.onError?.(parsed);
                                break;
                        }
                    } catch (e) {
                        console.warn('Failed to parse SSE data:', eventData, e);
                    }
                }
            }
        })
        .catch((err) => {
            if (err.name !== 'AbortError') {
                callbacks.onError?.({ message: err.message || 'Network error' });
            }
        });

    return controller;
}

// ============== Health Check ==============

export async function localHealthCheck(): Promise<{ status: string; service: string; version: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
}
