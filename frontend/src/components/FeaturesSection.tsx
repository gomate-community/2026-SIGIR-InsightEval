import { Card } from "@/components/ui/card"
import { 
  Layers, 
  GitCompare, 
  TrendingUp, 
  Network, 
  Zap, 
  Shield 
} from "lucide-react"

const features = [
  {
    icon: Layers,
    title: "Multi-dimensional Scoring",
    description: "Evaluates insight across three key dimensions: Synthesis, Critical Distance, and Abstraction Level.",
  },
  {
    icon: GitCompare,
    title: "Claim-Evidence Alignment",
    description: "Automatically pairs author viewpoints with supporting evidence from cited papers using RAG retrieval.",
  },
  {
    icon: TrendingUp,
    title: "Depth vs. Novelty",
    description: "Distinguishes genuine argumentative depth from surface-level innovation claims.",
  },
  {
    icon: Network,
    title: "Cross-document Analysis",
    description: "Identifies how authors synthesize insights across multiple referenced works.",
  },
  {
    icon: Zap,
    title: "Real-time Processing",
    description: "Rapid analysis pipeline that processes papers in seconds with streaming results.",
  },
  {
    icon: Shield,
    title: "Explainable AI",
    description: "Every score comes with detailed rationale explaining the AI's reasoning process.",
  },
]

export function FeaturesSection() {
  return (
    <section id="features" className="py-20 md:py-28 bg-muted/30">
      <div className="container px-4 md:px-6">
        <div className="mx-auto max-w-3xl text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight text-foreground md:text-4xl mb-4">
            Key Features
          </h2>
          <p className="text-lg text-muted-foreground text-pretty">
            InsightLens provides comprehensive tools for evaluating the depth and quality of academic argumentation
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <Card key={feature.title} className="p-6 transition-all hover:shadow-lg hover:-translate-y-1">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <feature.icon className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-2 text-lg font-semibold text-foreground">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
