import { Routes, Route } from 'react-router-dom'
import { FeaturesPage } from './pages/Features'
import { DemoPage } from './pages/Demo'

function App() {
  return (
    <Routes>
      <Route path="/" element={<FeaturesPage />} />
      <Route path="/demo" element={<DemoPage />} />
    </Routes>
  )
}

export default App
