'use client';

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Loader2, TrendingUp, TrendingDown, Minus } from "lucide-react"
import type { SamplePaper } from "@/lib/sample-papers"

interface SamplePaperSelectorProps {
  papers: SamplePaper[]
  onSelect: (paper: SamplePaper) => void
  isLoading: boolean
}

export function SamplePaperSelector({ papers, onSelect, isLoading }: SamplePaperSelectorProps) {
  const getInsightIcon = (level: string) => {
    switch (level) {
      case "Very High":
      case "High":
        return <TrendingUp className="h-3 w-3" />
      case "Low":
        return <TrendingDown className="h-3 w-3" />
      default:
        return <Minus className="h-3 w-3" />
    }
  }

  const getInsightColor = (level: string) => {
    switch (level) {
      case "Very High":
        return "bg-green-500/10 text-green-600 border-green-500/20"
      case "High":
        return "bg-primary/10 text-primary border-primary/20"
      case "Medium":
        return "bg-amber-500/10 text-amber-600 border-amber-500/20"
      case "Low":
        return "bg-muted text-muted-foreground border-muted"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {papers.map((paper) => (
        <Card
          key={paper.id}
          className="p-4 cursor-pointer transition-all hover:shadow-md hover:border-primary/30 group"
          onClick={() => !isLoading && onSelect(paper)}
        >
          <div className="flex flex-col h-full">
            <div className="flex items-start justify-between gap-2 mb-2">
              <Badge variant="outline" className="text-xs shrink-0">
                {paper.venue}
              </Badge>
              <Badge className={`text-xs shrink-0 flex items-center gap-1 ${getInsightColor(paper.insightLevel)}`}>
                {getInsightIcon(paper.insightLevel)}
                {paper.insightLevel}
              </Badge>
            </div>
            
            <h4 className="text-sm font-semibold text-foreground mb-1 line-clamp-2 group-hover:text-primary transition-colors">
              {paper.title}
            </h4>
            
            <p className="text-xs text-muted-foreground mb-3">{paper.authors} ({paper.year})</p>
            
            <p className="text-xs text-muted-foreground line-clamp-2 flex-1 mb-3">
              {paper.description}
            </p>

            <div className="flex items-center justify-between mt-auto pt-2 border-t">
              <div className="flex items-center gap-1">
                <span className="text-lg font-bold text-primary">{paper.overallScore.toFixed(1)}</span>
                <span className="text-xs text-muted-foreground">/5</span>
              </div>
              <Button 
                size="sm" 
                variant="ghost" 
                className="h-7 text-xs"
                disabled={isLoading}
              >
                {isLoading ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  "Analyze"
                )}
              </Button>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}
