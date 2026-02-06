# InsightLens - Academic Paper Insight Assessment

A Vite + React + TypeScript + Tailwind CSS demo system for evaluating argumentative depth in academic paper Introduction sections.

## Getting Started

### Prerequisites

- Node.js 18+ 
- pnpm (recommended) or npm

### Installation

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

## Project Structure

```
src/
├── components/
│   ├── ui/           # Reusable UI components (Button, Card, Badge, etc.)
│   ├── Header.tsx    # Navigation header
│   ├── Footer.tsx    # Footer component
│   ├── HeroSection.tsx
│   ├── PipelineSection.tsx
│   ├── FeaturesSection.tsx
│   ├── MetricsSection.tsx
│   ├── PaperSection.tsx
│   ├── SamplePaperSelector.tsx
│   └── InsightRadarChart.tsx
├── pages/
│   ├── Features.tsx  # Home page with features overview
│   └── Demo.tsx      # Interactive demo page
├── lib/
│   ├── utils.ts      # Utility functions
│   └── sample-papers.ts  # Sample paper data
├── App.tsx           # Main app with routing
├── main.tsx          # Entry point
└── index.css         # Global styles with Tailwind
```

## Features

- **Features Page**: Overview of the InsightLens system including pipeline, metrics, and paper information
- **Demo Page**: Interactive demonstration with:
  - Text file upload support
  - Pre-analyzed sample papers
  - Sentence-by-sentence analysis view
  - Radar chart visualization
  - Export to Markdown/JSON

## Tech Stack

- **Vite** - Build tool
- **React 18** - UI framework
- **TypeScript** - Type safety
- **React Router** - Client-side routing
- **Tailwind CSS** - Styling
- **Recharts** - Charts and visualizations
- **Lucide React** - Icons
- **Radix UI** - Accessible UI primitives

## License

MIT
