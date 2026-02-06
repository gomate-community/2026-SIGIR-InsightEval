import { Link } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { ArrowRight, Sparkles, FileSearch, Brain } from "lucide-react"

export function HeroSection() {
  return (
    <section className="relative overflow-hidden py-20 md:py-32">
      {/* Background decoration */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-1/4 left-1/4 h-96 w-96 rounded-full bg-primary/5 blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 h-96 w-96 rounded-full bg-chart-2/5 blur-3xl" />
      </div>

      <div className="container px-4 md:px-6">
        <div className="mx-auto max-w-4xl text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5">
            <Sparkles className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium text-primary">SIGIR 2026 Demo Track</span>
          </div>

          <h1 className="mb-6 text-4xl font-bold tracking-tight text-foreground md:text-5xl lg:text-6xl text-balance">
            Automated Insight Assessment for Academic Papers
          </h1>

          <p className="mx-auto mb-10 max-w-2xl text-lg text-muted-foreground leading-relaxed text-pretty">
            An intelligent system that quantifies the argumentative depth in paper Introduction sections, 
            distinguishing genuine scholarly insight from surface-level citation.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/demo">
              <Button size="lg" className="w-full sm:w-auto gap-2 text-base">
                Start Analysis
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Button variant="outline" size="lg" className="w-full sm:w-auto gap-2 text-base bg-transparent">
              Learn More
            </Button>
          </div>

          {/* Feature pills */}
          <div className="mt-16 flex flex-wrap items-center justify-center gap-3">
            {[
              { icon: FileSearch, label: "Structure Parsing" },
              { icon: Brain, label: "LLM-based Scoring" },
              { icon: Sparkles, label: "RAG Evidence Retrieval" },
            ].map((feature) => (
              <div
                key={feature.label}
                className="flex items-center gap-2 rounded-full border bg-card px-4 py-2 text-sm font-medium text-muted-foreground"
              >
                <feature.icon className="h-4 w-4 text-primary" />
                {feature.label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
