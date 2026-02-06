'use client';

import React from "react"

import { useState, useRef, useCallback } from "react"
import { Header } from "@/components/Header"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Upload,
  FileText,
  Loader2,
  ChevronRight,
  Download,
  FileDown,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  Quote,
  Brain,
  Lightbulb,
} from "lucide-react"
import { InsightRadarChart } from "@/components/InsightRadarChart"
import { SamplePaperSelector } from "@/components/SamplePaperSelector"
import { samplePapers, type SamplePaper } from "@/lib/sample-papers"
import {
  analyzePdf,
  analyzeText as apiAnalyzeText,
  type AnalysisResponse,
  calculateAverageScore
} from "@/lib/api"

// 前端展示用的句子类型
interface AnalyzedSentence {
  id: number
  text: string
  type: "context" | "citation" | "viewpoint"
  insightScore: number | null
  isHighInsight: boolean
  citationRef: string | null
  // 后端数据
  scores?: { synthesis: number; critical: number; abstraction: number }
  analysis?: string
  insightLevel?: "low" | "medium" | "high"
  evidence?: Array<{ quote: string; source: string; criteria: string }>
}

interface OverallAnalysis {
  synthesisScore: number
  criticalDistance: number
  abstractionLevel: number
  overallInsightScore: number
  insightLevel: string
  summary: string
  report?: {
    summary: string
    strengths: string[]
    weaknesses: string[]
    overall_score: number
  }
}

interface RationaleData {
  claim: string
  evidence: string
  rationale: string
  scores: {
    synthesis: number
    critical: number
    abstraction: number
  }
  keyInsights: string[]
}

// 将后端响应转换为前端展示格式
function transformApiResponse(response: AnalysisResponse): {
  sentences: AnalyzedSentence[]
  overallAnalysis: OverallAnalysis
} {
  const sentences: AnalyzedSentence[] = response.sentences.map((s) => {
    const avgScore = calculateAverageScore(s.scores)
    return {
      id: s.id,
      text: s.text,
      type: s.type,
      insightScore: avgScore,
      isHighInsight: s.insightLevel === "high",
      citationRef: s.source || null,
      scores: s.scores,
      analysis: s.analysis,
      insightLevel: s.insightLevel,
      evidence: s.evidence,
    }
  })

  const avgSynthesis = sentences.length > 0
    ? sentences.reduce((acc, s) => acc + (s.scores?.synthesis ?? 0), 0) / sentences.length
    : 0
  const avgCritical = sentences.length > 0
    ? sentences.reduce((acc, s) => acc + (s.scores?.critical ?? 0), 0) / sentences.length
    : 0
  const avgAbstraction = sentences.length > 0
    ? sentences.reduce((acc, s) => acc + (s.scores?.abstraction ?? 0), 0) / sentences.length
    : 0

  const overallAnalysis: OverallAnalysis = {
    synthesisScore: Math.round(avgSynthesis * 10) / 10,
    criticalDistance: Math.round(avgCritical * 10) / 10,
    abstractionLevel: Math.round(avgAbstraction * 10) / 10,
    overallInsightScore: response.overallScore,
    insightLevel: response.report?.overall_score
      ? (response.report.overall_score >= 7 ? "High" : response.report.overall_score >= 4 ? "Medium" : "Low")
      : "Medium",
    summary: response.summary,
    report: response.report,
  }

  return { sentences, overallAnalysis }
}

export function DemoPage() {
  const [selectedSentence, setSelectedSentence] = useState<number | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isParsing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [paperTitle, setPaperTitle] = useState<string>("")
  const [analysisStep, setAnalysisStep] = useState<string>("")
  const [progressPercent, setProgressPercent] = useState(0)

  const [analyzedSentences, setAnalyzedSentences] = useState<AnalyzedSentence[]>([])
  const [overallAnalysis, setOverallAnalysis] = useState<OverallAnalysis | null>(null)
  const [selectedRationale, setSelectedRationale] = useState<RationaleData | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // Handle file upload - call real backend API
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setError(null)
    setIsAnalyzing(true)
    setProgressPercent(0)
    setPaperTitle(file.name.replace(/\.[^/.]+$/, ""))

    try {
      // 根据文件类型选择处理方式
      const isPdf = file.name.toLowerCase().endsWith('.pdf')

      if (isPdf) {
        // PDF 文件 - 使用后端 PDF 分析接口
        setAnalysisStep("正在上传 PDF 文件...")
        setProgressPercent(10)

        const response = await analyzePdf(file, (step, percent) => {
          setAnalysisStep(step)
          setProgressPercent(percent)
        })

        setAnalysisStep("处理完成")
        setProgressPercent(100)

        // 转换响应格式
        const { sentences, overallAnalysis } = transformApiResponse(response)
        setAnalyzedSentences(sentences)
        setOverallAnalysis(overallAnalysis)
        if (response.paperTitle) {
          setPaperTitle(response.paperTitle)
        }
        setSelectedSentence(sentences.length > 0 ? 1 : null)
      } else {
        // 文本文件 - 读取并调用文本分析接口
        setAnalysisStep("正在读取文件...")
        setProgressPercent(10)

        const text = await file.text()

        setAnalysisStep("正在分析文本...")
        setProgressPercent(30)

        const response = await apiAnalyzeText(text, file.name.replace(/\.[^/.]+$/, ""))

        setAnalysisStep("处理完成")
        setProgressPercent(100)

        const { sentences, overallAnalysis } = transformApiResponse(response)
        setAnalyzedSentences(sentences)
        setOverallAnalysis(overallAnalysis)
        setSelectedSentence(sentences.length > 0 ? 1 : null)
      }
    } catch (err) {
      console.error("Analysis error:", err)
      setError(err instanceof Error ? err.message : "分析失败，请稍后重试")
    } finally {
      setIsAnalyzing(false)
      setAnalysisStep("")
    }
  }

  const handleSamplePaperSelect = (paper: SamplePaper) => {
    setError(null)
    setPaperTitle(paper.title)

    if (paper.preAnalysis) {
      setAnalyzedSentences(paper.preAnalysis.sentences)
      setOverallAnalysis(paper.preAnalysis.overallAnalysis)
      setSelectedSentence(1)
      setSelectedRationale(null)
    }
  }

  const loadRationale = useCallback(
    (sentenceId: number) => {
      const sentence = analyzedSentences.find((s) => s.id === sentenceId)
      if (!sentence || sentence.insightScore === null) {
        setSelectedRationale(null)
        return
      }

      // 使用后端返回的 analysis 和 evidence，如果有的话
      const evidenceText = sentence.evidence && sentence.evidence.length > 0
        ? sentence.evidence.map(e => `[${e.source}]: ${e.quote}`).join("\n")
        : "Based on the surrounding context and cited sources."

      setSelectedRationale({
        claim: sentence.text,
        evidence: evidenceText,
        rationale: sentence.analysis || `This sentence demonstrates ${sentence.isHighInsight ? "high" : "moderate"} insight by ${sentence.type === "viewpoint" ? "presenting an original argument that goes beyond the cited sources" : sentence.type === "citation" ? "referencing prior work with some synthesis" : "providing background context"}. The insight score of ${sentence.insightScore?.toFixed(1)} reflects the degree to which it adds new understanding.`,
        scores: {
          synthesis: sentence.scores?.synthesis ?? sentence.insightScore ?? 0,
          critical: sentence.scores?.critical ?? (sentence.insightScore ?? 0) * (sentence.isHighInsight ? 1.1 : 0.9),
          abstraction: sentence.scores?.abstraction ?? (sentence.insightScore ?? 0) * (sentence.type === "viewpoint" ? 1.05 : 0.95),
        },
        keyInsights: [
          sentence.isHighInsight ? "High argumentative depth" : "Moderate contribution",
          `Sentence type: ${sentence.type}`,
          sentence.citationRef ? `References: ${sentence.citationRef}` : "Original viewpoint",
        ],
      })
    },
    [analyzedSentences]
  )

  const handleSentenceSelect = (id: number) => {
    setSelectedSentence(id)
    loadRationale(id)
  }

  const handleExport = (format: "markdown" | "json") => {
    if (!analyzedSentences.length || !overallAnalysis) return

    const data = {
      paperTitle,
      sentences: analyzedSentences,
      overallAnalysis,
      exportedAt: new Date().toISOString(),
    }

    let content: string
    let filename: string
    let mimeType: string

    if (format === "json") {
      content = JSON.stringify(data, null, 2)
      filename = `insight-analysis-${Date.now()}.json`
      mimeType = "application/json"
    } else {
      content = `# Insight Analysis Report

## Paper: ${paperTitle}

### Overall Scores
- **Overall Insight Score:** ${overallAnalysis.overallInsightScore.toFixed(1)}/10
- **Insight Level:** ${overallAnalysis.insightLevel}
- **Synthesis Score:** ${overallAnalysis.synthesisScore.toFixed(1)}/5
- **Critical Distance:** ${overallAnalysis.criticalDistance.toFixed(1)}/5
- **Abstraction Level:** ${overallAnalysis.abstractionLevel.toFixed(1)}/5

### Summary
${overallAnalysis.summary}

${overallAnalysis.report ? `### Detailed Report

**Strengths:**
${overallAnalysis.report.strengths.map(s => `- ${s}`).join('\n')}

**Weaknesses:**
${overallAnalysis.report.weaknesses.map(w => `- ${w}`).join('\n')}
` : ''}

### Sentence-by-Sentence Analysis

| # | Type | Score | Text |
|---|------|-------|------|
${analyzedSentences.map(s => `| ${s.id} | ${s.type} | ${s.insightScore?.toFixed(1) ?? 'N/A'} | ${s.text.slice(0, 60)}... |`).join('\n')}

---
*Generated by InsightLens on ${new Date().toLocaleDateString()}*
`
      filename = `insight-analysis-${Date.now()}.md`
      mimeType = "text/markdown"
    }

    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  const handleReset = () => {
    setAnalyzedSentences([])
    setOverallAnalysis(null)
    setSelectedSentence(null)
    setSelectedRationale(null)
    setPaperTitle("")
    setError(null)
    setAnalysisStep("")
    setProgressPercent(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case "context":
        return "text-muted-foreground"
      case "citation":
        return "text-blue-600"
      case "viewpoint":
        return "text-foreground"
      default:
        return "text-foreground"
    }
  }

  const getInsightColor = (insight: number | null) => {
    if (insight === null) return "bg-transparent"
    if (insight >= 4) return "bg-primary/20 border-l-4 border-l-primary"
    if (insight >= 3) return "bg-amber-500/10 border-l-4 border-l-amber-500"
    return "bg-muted/50 border-l-4 border-l-muted-foreground/30"
  }

  const highInsightCount = analyzedSentences.filter((s) => s.isHighInsight).length
  const selectedSentenceData = analyzedSentences.find((s) => s.id === selectedSentence)

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="py-8 md:py-12">
        <div className="container px-4 md:px-6">
          {/* Page Header */}
          <div className="mx-auto max-w-4xl text-center mb-10">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
              <Sparkles className="h-4 w-4" />
              Interactive Demo
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground md:text-4xl mb-4 text-balance">
              Paper Insight Analysis
            </h1>
            <p className="text-lg text-muted-foreground text-pretty max-w-2xl mx-auto">
              Upload your paper (PDF or text) to see how InsightLens evaluates the argumentative depth
              of Introduction sections using AI analysis
            </p>
          </div>

          {/* Upload area */}
          <div className="mx-auto max-w-5xl mb-8">
            <Card className="border-2 border-dashed border-border/60 bg-muted/20 p-8">
              <div className="flex flex-col items-center text-center">
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10">
                  <Upload className="h-6 w-6 text-primary" />
                </div>
                <h3 className="mb-2 text-lg font-semibold text-foreground">Upload Paper</h3>
                <p className="mb-4 text-sm text-muted-foreground">Upload a PDF or text file with your paper, or select a sample below</p>

                {error && (
                  <div className="mb-4 flex items-center gap-2 text-sm text-destructive bg-destructive/10 px-4 py-2 rounded-lg">
                    <AlertCircle className="h-4 w-4" />
                    {error}
                  </div>
                )}

                <div className="flex flex-wrap justify-center gap-3 mb-6">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                  />
                  <Button
                    variant="outline"
                    className="gap-2 bg-transparent"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isParsing || isAnalyzing}
                  >
                    {isParsing ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Parsing...
                      </>
                    ) : (
                      <>
                        <FileText className="h-4 w-4" />
                        Browse Files
                      </>
                    )}
                  </Button>

                  {analyzedSentences.length > 0 && (
                    <Button variant="outline" className="gap-2 bg-transparent" onClick={handleReset}>
                      <RefreshCw className="h-4 w-4" />
                      Reset
                    </Button>
                  )}
                </div>

                {/* Sample Paper Selector */}
                <div className="w-full border-t pt-6">
                  <p className="text-sm text-muted-foreground mb-4">Or try one of our sample papers with pre-analyzed data:</p>
                  <SamplePaperSelector papers={samplePapers} onSelect={handleSamplePaperSelect} isLoading={isAnalyzing} />
                </div>
              </div>
            </Card>
          </div>

          {/* Loading state */}
          {isAnalyzing && (
            <div className="mx-auto max-w-5xl mb-8">
              <Card className="p-8">
                <div className="flex flex-col items-center text-center">
                  <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
                  <h3 className="text-lg font-semibold text-foreground mb-2">Analyzing Paper...</h3>
                  <p className="text-sm text-muted-foreground mb-2">
                    {analysisStep || "Evaluating argumentative depth of your paper's introduction"}
                  </p>
                  <div className="w-full max-w-xs mt-4">
                    <Progress value={progressPercent} className="h-2" />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">{progressPercent}%</p>
                </div>
              </Card>
            </div>
          )}

          {/* Main demo interface */}
          {analyzedSentences.length > 0 && overallAnalysis && !isAnalyzing && (
            <>
              {/* Paper title header */}
              {paperTitle && (
                <div className="mx-auto max-w-7xl mb-6">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0" />
                      <h3 className="font-semibold text-foreground truncate max-w-lg">{paperTitle}</h3>
                      <Badge variant="secondary" className="text-xs shrink-0">
                        {overallAnalysis.insightLevel}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="gap-2 bg-transparent"
                        onClick={() => handleExport("markdown")}
                      >
                        <FileDown className="h-4 w-4" />
                        Export MD
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="gap-2 bg-transparent"
                        onClick={() => handleExport("json")}
                      >
                        <Download className="h-4 w-4" />
                        Export JSON
                      </Button>
                    </div>
                  </div>
                </div>
              )}

              <div className="mx-auto max-w-7xl">
                <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
                  {/* Left: Document Stream */}
                  <Card className="p-0 overflow-hidden">
                    <div className="border-b bg-muted/30 px-6 py-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold text-foreground">Document Stream</h3>
                          <p className="text-sm text-muted-foreground">Introduction Section - Click to analyze</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary" className="text-xs">
                            {analyzedSentences.length} sentences
                          </Badge>
                          {highInsightCount > 0 && (
                            <Badge className="text-xs bg-primary/10 text-primary border-primary/20">
                              {highInsightCount} high insight
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="p-6 space-y-3 max-h-[600px] overflow-y-auto">
                      {analyzedSentences.map((sentence) => (
                        <div
                          key={sentence.id}
                          onClick={() => handleSentenceSelect(sentence.id)}
                          className={`p-4 rounded-lg cursor-pointer transition-all ${getInsightColor(sentence.insightScore)} ${selectedSentence === sentence.id ? "ring-2 ring-primary ring-offset-2" : "hover:bg-muted/50"
                            }`}
                        >
                          <div className="flex items-start gap-3">
                            <div className="flex items-center gap-2 shrink-0">
                              <span className="text-xs font-medium text-muted-foreground w-5">{sentence.id}</span>
                              {sentence.insightScore !== null && (
                                <div className="flex items-center justify-center h-6 w-6 rounded-full bg-primary/10 text-xs font-semibold text-primary">
                                  {sentence.insightScore.toFixed(1)}
                                </div>
                              )}
                            </div>
                            <div className="flex-1">
                              <p className={`text-sm leading-relaxed ${getTypeColor(sentence.type)}`}>
                                {sentence.citationRef && (
                                  <span className="text-primary font-medium">{sentence.citationRef} </span>
                                )}
                                {sentence.text}
                              </p>
                              <div className="mt-2 flex items-center gap-2">
                                <Badge variant="outline" className="text-xs capitalize">
                                  {sentence.type}
                                </Badge>
                                {sentence.isHighInsight && (
                                  <Badge className="text-xs bg-primary/10 text-primary border-0">High Insight</Badge>
                                )}
                              </div>
                            </div>
                            <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0 mt-1" />
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>

                  {/* Right: Analysis Panel */}
                  <Card className="p-0 overflow-hidden">
                    <div className="border-b bg-muted/30 px-6 py-4">
                      <h3 className="font-semibold text-foreground">Insight Analysis</h3>
                      <p className="text-sm text-muted-foreground">
                        {selectedSentence ? `Sentence #${selectedSentence}` : "Select a sentence to analyze"}
                      </p>
                    </div>

                    <Tabs defaultValue="alignment" className="p-6">
                      <TabsList className="grid w-full grid-cols-3 mb-6">
                        <TabsTrigger value="alignment">Alignment</TabsTrigger>
                        <TabsTrigger value="scores">Scores</TabsTrigger>
                        <TabsTrigger value="rationale">AI Rationale</TabsTrigger>
                      </TabsList>

                      <TabsContent value="alignment" className="space-y-4">
                        {selectedSentenceData ? (
                          <>
                            <div className="space-y-4">
                              <div className="rounded-lg border p-4">
                                <div className="flex items-center gap-2 mb-2">
                                  <Quote className="h-4 w-4 text-primary" />
                                  <span className="text-xs font-semibold text-muted-foreground uppercase">Claim</span>
                                </div>
                                <p className="text-sm text-foreground">{selectedSentenceData.text}</p>
                              </div>

                              <div className="rounded-lg border border-dashed p-4 bg-muted/20">
                                <div className="flex items-center gap-2 mb-2">
                                  <Brain className="h-4 w-4 text-chart-2" />
                                  <span className="text-xs font-semibold text-muted-foreground uppercase">Evidence</span>
                                </div>
                                {selectedSentenceData.evidence && selectedSentenceData.evidence.length > 0 ? (
                                  <div className="space-y-2">
                                    {selectedSentenceData.evidence.map((e, i) => (
                                      <div key={i} className="text-sm text-muted-foreground">
                                        <span className="font-medium text-foreground">[{e.source}]:</span> {e.quote}
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <p className="text-sm text-muted-foreground">
                                    {selectedSentenceData.citationRef
                                      ? `Referenced from ${selectedSentenceData.citationRef}`
                                      : "Original viewpoint without direct citation"}
                                  </p>
                                )}
                              </div>
                            </div>
                          </>
                        ) : (
                          <div className="text-center py-12 text-muted-foreground">
                            <Quote className="h-12 w-12 mx-auto mb-4 opacity-20" />
                            <p>Select a sentence to view alignment details</p>
                          </div>
                        )}
                      </TabsContent>

                      <TabsContent value="scores" className="space-y-6">
                        {overallAnalysis && (
                          <>
                            <div className="flex justify-center">
                              <InsightRadarChart
                                scores={{
                                  synthesis: overallAnalysis.synthesisScore,
                                  critical: overallAnalysis.criticalDistance,
                                  abstraction: overallAnalysis.abstractionLevel,
                                }}
                              />
                            </div>
                            <div className="space-y-3">
                              {[
                                { label: "Synthesis Score", value: overallAnalysis.synthesisScore },
                                { label: "Critical Distance", value: overallAnalysis.criticalDistance },
                                { label: "Abstraction Level", value: overallAnalysis.abstractionLevel },
                              ].map((item) => (
                                <div key={item.label} className="flex items-center justify-between">
                                  <span className="text-sm text-muted-foreground">{item.label}</span>
                                  <div className="flex items-center gap-2">
                                    <div className="w-24 h-2 rounded-full bg-muted overflow-hidden">
                                      <div
                                        className="h-full bg-primary rounded-full transition-all"
                                        style={{ width: `${(item.value / 5) * 100}%` }}
                                      />
                                    </div>
                                    <span className="text-sm font-semibold text-foreground w-8">
                                      {item.value.toFixed(1)}
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                            <div className="pt-4 border-t">
                              <div className="flex items-center justify-between">
                                <span className="font-semibold text-foreground">Overall Insight</span>
                                <div className="flex items-center gap-2">
                                  <span className="text-2xl font-bold text-primary">
                                    {overallAnalysis.overallInsightScore.toFixed(1)}
                                  </span>
                                  <span className="text-muted-foreground">/10</span>
                                </div>
                              </div>
                            </div>
                          </>
                        )}
                      </TabsContent>

                      <TabsContent value="rationale" className="space-y-4">
                        {selectedRationale ? (
                          <>
                            <div className="rounded-lg border p-4 bg-primary/5">
                              <div className="flex items-center gap-2 mb-2">
                                <Lightbulb className="h-4 w-4 text-primary" />
                                <span className="text-xs font-semibold text-muted-foreground uppercase">AI Assessment</span>
                              </div>
                              <p className="text-sm text-foreground leading-relaxed">
                                {selectedRationale.rationale}
                              </p>
                            </div>

                            <div className="space-y-2">
                              <span className="text-xs font-semibold text-muted-foreground uppercase">Key Observations</span>
                              <ul className="space-y-1">
                                {selectedRationale.keyInsights.map((insight, i) => (
                                  <li key={i} className="flex items-center gap-2 text-sm text-muted-foreground">
                                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                                    {insight}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </>
                        ) : (
                          <div className="text-center py-12 text-muted-foreground">
                            <Brain className="h-12 w-12 mx-auto mb-4 opacity-20" />
                            <p>Select a sentence with an insight score to view AI rationale</p>
                          </div>
                        )}
                      </TabsContent>
                    </Tabs>
                  </Card>
                </div>
              </div>

              {/* Summary Card */}
              <div className="mx-auto max-w-7xl mt-6">
                <Card className="p-6">
                  <h3 className="font-semibold text-foreground mb-2">Analysis Summary</h3>
                  <p className="text-sm text-muted-foreground mb-4">{overallAnalysis.summary}</p>

                  {/* Report Details */}
                  {overallAnalysis.report && (
                    <div className="grid md:grid-cols-2 gap-6 mt-4 pt-4 border-t">
                      <div>
                        <h4 className="text-sm font-semibold text-green-600 mb-2 flex items-center gap-2">
                          <CheckCircle2 className="h-4 w-4" />
                          Strengths
                        </h4>
                        <ul className="space-y-1">
                          {overallAnalysis.report.strengths.map((s, i) => (
                            <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                              <span className="h-1.5 w-1.5 rounded-full bg-green-500 mt-1.5 shrink-0" />
                              {s}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-amber-600 mb-2 flex items-center gap-2">
                          <AlertCircle className="h-4 w-4" />
                          Areas for Improvement
                        </h4>
                        <ul className="space-y-1">
                          {overallAnalysis.report.weaknesses.map((w, i) => (
                            <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                              <span className="h-1.5 w-1.5 rounded-full bg-amber-500 mt-1.5 shrink-0" />
                              {w}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </Card>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
