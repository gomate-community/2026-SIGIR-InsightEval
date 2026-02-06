import { Header } from "@/components/Header"
import { HeroSection } from "@/components/HeroSection"
import { PipelineSection } from "@/components/PipelineSection"
import { MetricsSection } from "@/components/MetricsSection"
import { FeaturesSection } from "@/components/FeaturesSection"
import { PaperSection } from "@/components/PaperSection"
import { Footer } from "@/components/Footer"

export function FeaturesPage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <HeroSection />
        <PipelineSection />
        <MetricsSection />
        <FeaturesSection />
        <PaperSection />
      </main>
      <Footer />
    </div>
  )
}
