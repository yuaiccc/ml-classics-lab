import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "@/pages/Home";
import ExperimentDetail from "@/pages/ExperimentDetail";
import AlgorithmLab from "@/pages/AlgorithmLab";

export default function App() {
  return (
    <Router>
      <div className="bg-grid min-h-screen">
        <Routes>
          <Route path="/" element={<AlgorithmLab />} />
          <Route path="/lab" element={<AlgorithmLab />} />
          <Route path="/dashboard" element={<Home />} />
          <Route path="/experiment/:id" element={<ExperimentDetail />} />
        </Routes>
      </div>
    </Router>
  );
}
