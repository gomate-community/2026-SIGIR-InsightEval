import { Routes, Route } from 'react-router-dom'
import { FeaturesPage } from './pages/Features'
import { DemoPage } from './pages/Demo'
import { LocalAnalysisPage } from './pages/LocalAnalysis'

function App() {
  return (
    <Routes>
      <Route path="/" element={<FeaturesPage />} />
      <Route path="/demo" element={<DemoPage />} />
      <Route path="/local" element={<LocalAnalysisPage />} />
    </Routes>
  )
}

export default App
