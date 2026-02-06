import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { FileText, Download, ExternalLink, Users, Calendar } from "lucide-react"

export function PaperSection() {
  return (
    <section id="paper" className="py-20 md:py-28 bg-muted/30">
      <div className="container px-4 md:px-6">
        <div className="mx-auto max-w-4xl">
          <Card className="p-8 md:p-12">
            <div className="flex flex-col md:flex-row gap-8">
              {/* Paper preview */}
              <div className="w-full md:w-48 shrink-0">
                <div className="aspect-[3/4] rounded-lg bg-gradient-to-br from-primary/20 to-chart-2/20 flex items-center justify-center border">
                  <FileText className="h-16 w-16 text-primary/40" />
                </div>
              </div>

              {/* Paper info */}
              <div className="flex-1">
                <Badge className="mb-4 bg-primary/10 text-primary border-0">SIGIR 2026 Demo Track</Badge>
                
                <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4 text-balance">
                  InsightLens: Automated Argumentative Depth Assessment for Academic Paper Introductions
                </h2>

                <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mb-6">
                  <div className="flex items-center gap-2">
                    <Users className="h-4 w-4" />
                    <span>Anonymous Authors</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    <span>2026</span>
                  </div>
                </div>

                <p className="text-muted-foreground leading-relaxed mb-6">
                  We present InsightLens, a novel system for automatically evaluating the argumentative depth 
                  in academic paper Introduction sections. Unlike existing tools that focus on citation networks 
                  or novelty detection, InsightLens quantifies the "Insight Gain" — the degree to which authors 
                  synthesize, critique, and abstract beyond their cited sources. Our system employs a four-phase 
                  pipeline combining structure parsing, RAG-based evidence retrieval, LLM-powered scoring, and 
                  interactive visualization.
                </p>

                <div className="flex flex-wrap gap-3 mb-8">
                  {["Insight Assessment", "LLM", "RAG", "Academic Writing", "NLP"].map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      #{tag.toLowerCase().replace(" ", "-")}
                    </Badge>
                  ))}
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button className="gap-2">
                    <Download className="h-4 w-4" />
                    Download PDF
                  </Button>
                  <Button variant="outline" className="gap-2 bg-transparent">
                    <ExternalLink className="h-4 w-4" />
                    View on arXiv
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </section>
  )
}
