import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const metrics = [
  {
    name: "Synthesis Score",
    description: "Measures how well the author connects and integrates insights from multiple cited works.",
    lowExample: {
      text: "Paper A did X. Paper B did Y.",
      label: "Low Insight",
    },
    highExample: {
      text: "Although Paper A solved X, Paper B's results on Y suggest X fails under condition Z.",
      label: "High Insight",
    },
    formula: "Cross-document semantic linking density",
  },
  {
    name: "Critical Distance",
    description: "Evaluates whether the author identifies limitations, challenges, or gaps in cited work.",
    lowExample: {
      text: "[1] proposed the Transformer architecture.",
      label: "Low Insight",
    },
    highExample: {
      text: "Despite [1]'s success, the quadratic complexity limits practical long-context deployment.",
      label: "High Insight",
    },
    formula: "Presence of critique markers + limitation analysis",
  },
  {
    name: "Abstraction Level",
    description: "Assesses whether viewpoints operate at a higher conceptual level than the cited evidence.",
    lowExample: {
      text: "The model achieved 95% accuracy on the benchmark.",
      label: "Descriptive",
    },
    highExample: {
      text: "This pattern suggests a fundamental trade-off between efficiency and expressiveness.",
      label: "Generalization",
    },
    formula: "Conceptual hierarchy gap measurement",
  },
]

export function MetricsSection() {
  return (
    <section className="py-20 md:py-28">
      <div className="container px-4 md:px-6">
        <div className="mx-auto max-w-3xl text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight text-foreground md:text-4xl mb-4">
            Insight Metrics
          </h2>
          <p className="text-lg text-muted-foreground text-pretty">
            Our three-dimensional framework for quantifying argumentative depth
          </p>
        </div>

        <div className="space-y-8 max-w-4xl mx-auto">
          {metrics.map((metric, index) => (
            <Card key={metric.name} className="p-6 md:p-8">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-xl font-bold text-primary shrink-0">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-foreground mb-2">{metric.name}</h3>
                  <p className="text-muted-foreground mb-6">{metric.description}</p>

                  <div className="grid gap-4 md:grid-cols-2 mb-4">
                    <div className="rounded-lg bg-muted/50 p-4">
                      <Badge variant="outline" className="mb-2 text-xs">{metric.lowExample.label}</Badge>
                      <p className="text-sm text-muted-foreground italic">"{metric.lowExample.text}"</p>
                    </div>
                    <div className="rounded-lg bg-primary/5 border border-primary/20 p-4">
                      <Badge className="mb-2 text-xs bg-primary/10 text-primary border-0">{metric.highExample.label}</Badge>
                      <p className="text-sm text-foreground italic">"{metric.highExample.text}"</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="font-semibold">Computation:</span>
                    <code className="rounded bg-muted px-2 py-0.5">{metric.formula}</code>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
