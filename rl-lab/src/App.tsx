import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AlgorithmLab from "@/pages/AlgorithmLab";

export default function App() {
  return (
    <Router>
      <div className="bg-grid min-h-screen">
        <Routes>
          <Route path="/" element={<AlgorithmLab />} />
          <Route path="/lab" element={<AlgorithmLab />} />
        </Routes>
      </div>
    </Router>
  );
}
