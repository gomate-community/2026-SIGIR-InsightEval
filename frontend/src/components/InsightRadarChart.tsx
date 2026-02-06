import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts"

interface InsightRadarChartProps {
  scores: {
    synthesis: number
    critical: number
    abstraction: number
  }
}

export function InsightRadarChart({ scores }: InsightRadarChartProps) {
  const data = [
    { metric: "Synthesis", score: scores.synthesis, fullMark: 5 },
    { metric: "Critical", score: scores.critical, fullMark: 5 },
    { metric: "Abstraction", score: scores.abstraction, fullMark: 5 },
  ]

  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart data={data}>
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'hsl(var(--card))', 
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
            fontSize: '12px'
          }} 
        />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }} />
        <PolarGrid gridType="polygon" stroke="hsl(var(--border))" />
        <Radar
          name="Score"
          dataKey="score"
          stroke="hsl(var(--primary))"
          fill="hsl(var(--primary))"
          fillOpacity={0.3}
          strokeWidth={2}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}
