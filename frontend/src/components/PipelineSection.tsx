import { FileDown, Search, BarChart3, FileOutput, ArrowRight } from "lucide-react"

const phases = [
  {
    number: "I",
    title: "Structure Parsing",
    subtitle: "& Extraction",
    color: "bg-blue-500/10 border-blue-500/30",
    iconColor: "text-blue-600",
    items: [
      "PDF Download + Text Extraction",
      "Introduction Section Extraction",
      "Sentence-Level Segmentation",
      "Semantic Unit Classification",
    ],
    description: "Context / Citation / Viewpoint",
    icon: FileDown,
  },
  {
    number: "II",
    title: "Evidence Retrieval",
    subtitle: "& Alignment",
    color: "bg-emerald-500/10 border-emerald-500/30",
    iconColor: "text-emerald-600",
    items: [
      "Citation Extraction & Resolution",
      "RAG-based Semantic Search",
      "Top-k Candidate Set Selection",
      "Evidence Alignment & Pairing",
    ],
    description: "Wispaper / Abstract / Conclusion",
    icon: Search,
  },
  {
    number: "III",
    title: "Insight Scoring",
    subtitle: "& Analysis",
    color: "bg-amber-500/10 border-amber-500/30",
    iconColor: "text-amber-600",
    items: [
      "Synthesis Score (Cross-document)",
      "Critical Distance (Limitation)",
      "Abstraction Level (Higher-level)",
      "Overall Insight Assessment",
    ],
    description: "LLM-based CoT Evaluation",
    icon: BarChart3,
  },
  {
    number: "IV",
    title: "Render",
    subtitle: "& Visualization",
    color: "bg-indigo-500/10 border-indigo-500/30",
    iconColor: "text-indigo-600",
    items: [
      "Report Rendering (JSON)",
      "Interactive Visualization",
      "Hierarchical Indentation",
      "Export to MD/PDF/Website",
    ],
    description: "Highlights / Radar Chart",
    icon: FileOutput,
  },
]

export function PipelineSection() {
  return (
    <section id="pipeline" className="py-20 md:py-28 bg-muted/30">
      <div className="container px-4 md:px-6">
        <div className="mx-auto max-w-3xl text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight text-foreground md:text-4xl mb-4">
            System Pipeline
          </h2>
          <p className="text-lg text-muted-foreground text-pretty">
            A four-phase automated workflow that transforms raw PDF papers into quantified insight assessments
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
                      Phase {phase.number}
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

                <div className="pt-4 border-t border-border/50">
                  <span className="text-xs font-medium text-primary">{phase.description}</span>
                </div>
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
            <span className="text-muted-foreground">PDF / URL</span>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground rotate-90 sm:rotate-0" />
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/30">
            <span className="font-semibold text-emerald-600">OUTPUT</span>
            <span className="text-muted-foreground">MD / PDF / Website</span>
          </div>
        </div>
      </div>
    </section>
  )
}
