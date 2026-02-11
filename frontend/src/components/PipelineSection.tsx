import { FileText, Brain, BarChart3, FileCheck, ArrowRight } from "lucide-react"

const phases = [
  {
    number: "I",
    title: "Opinion Sentence Extraction",
    subtitle: "PDF Parsing & Classification",
    color: "bg-blue-500/10 border-blue-500/30",
    iconColor: "text-blue-600",
    items: [
      "PDF Parsing via MinerU",
      "Citation Marker Detection",
      "Sentence-Level Segmentation",
      "Citation / Opinion Classification",
    ],
    // description: "Input: Paper PDF → Output: Classified Sentences",
    icon: FileText,
  },
  {
    number: "II",
    title: "Evidence Agentic Retrieval",
    subtitle: "RAG-based Alignment",
    color: "bg-emerald-500/10 border-emerald-500/30",
    iconColor: "text-emerald-600",
    items: [
      "LLM-based Viewpoint Analysising",
      "Citation Resolution to References",
      "Semantic Retrieval from  PDFs",
      "Evidence-Opinion Pair Construction",
    ],
    // description: "Input: Opinion Sentences → Output: (Opinion, Evidence) Pairs",
    icon: Brain,
  },
  {
    number: "III",
    title: "Multi-Dimensional Scoring",
    subtitle: "LLM-based CoT Evaluation",
    color: "bg-amber-500/10 border-amber-500/30",
    iconColor: "text-amber-600",
    items: [
      "Depth Score (1–5): Beyond Evidence",
      "Breadth Score (1–5): Cross-source Synthesis",
      "Height Score (1–5): Abstraction Level",
      "Composite Insight Score Calculation",
    ],
    // description: "Input: (Opinion, Evidence) → Output: Dimensional Scores",
    icon: BarChart3,
  },
  {
    number: "IV",
    title: "Report Synthesis",
    subtitle: "Paper-Level Assessment",
    color: "bg-indigo-500/10 border-indigo-500/30",
    iconColor: "text-indigo-600",
    items: [
      "Score Aggregation (Mean / Median)",
      "Strengths & Weaknesses Analysis",
      "Introduction + Scores →  Summary",
      "Overall Insightfulness Report",
    ],
    // description: "Input: All Scores → Output: Insight Report",
    icon: FileCheck,
  },
]

export function PipelineSection() {
  return (
    <section id="pipeline" className="py-20 md:py-28 bg-muted/30">
      <div className="container px-4 md:px-6">
        <div className="mx-auto max-w-3xl text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight text-foreground md:text-4xl mb-4">
            Evaluation Pipeline
          </h2>
          <p className="text-lg text-muted-foreground text-pretty">
            A four-stage automated pipeline that extracts opinion sentences from scientific papers,
            retrieves reference evidence, scores insightfulness along multiple dimensions,
            and synthesizes a comprehensive evaluation report.
          </p>
        </div>

        {/* Pipeline visualization */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {phases.map((phase, index) => (
            <div key={phase.number} className="relative">
              <div
                className={`rounded-xl border-2 ${phase.color} p-6 h-full bg-card transition-all hover:shadow-lg hover:-translate-y-1`}
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className={`flex h-10 w-10 items-center justify-center rounded-lg bg-card border ${phase.iconColor}`}>
                    <phase.icon className="h-5 w-5" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      Stage {phase.number}
                    </div>
                    <div className="font-semibold text-foreground">{phase.title}</div>
                  </div>
                </div>

                <p className="text-sm font-medium text-muted-foreground mb-4">{phase.subtitle}</p>

                <ul className="space-y-2 mb-4">
                  {phase.items.map((item) => (
                    <li key={item} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary/60 shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>

                {/*<div className="pt-4 border-t border-border/50">*/}
                {/*  <span className="text-xs font-medium text-primary">{phase.description}</span>*/}
                {/*</div>*/}
              </div>

              {/* Arrow connector */}
              {index < phases.length - 1 && (
                <div className="hidden lg:flex absolute top-1/2 -right-3 z-10 h-6 w-6 items-center justify-center rounded-full bg-background border border-border">
                  <ArrowRight className="h-3 w-3 text-muted-foreground" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Input/Output indicators */}
        <div className="mt-12 flex flex-col sm:flex-row items-center justify-center gap-8 text-sm">
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/30">
            <span className="font-semibold text-blue-600">INPUT</span>
            <span className="text-muted-foreground">Paper PDF + Reference PDFs</span>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground rotate-90 sm:rotate-0" />
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/30">
            <span className="font-semibold text-emerald-600">OUTPUT</span>
            <span className="text-muted-foreground">Insightfulness Report + Scores</span>
          </div>
        </div>
      </div>
    </section>
  )
}
