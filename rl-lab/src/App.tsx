import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "@/pages/Home";
import ExperimentDetail from "@/pages/ExperimentDetail";
import AlgorithmLab from "@/pages/AlgorithmLab";

export default function App() {
  return (
    <Router>
      <div className="bg-grid min-h-screen">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/experiment/:id" element={<ExperimentDetail />} />
          <Route path="/lab" element={<AlgorithmLab />} />
        </Routes>
      </div>
    </Router>
  );
}
