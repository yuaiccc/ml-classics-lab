import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "@/pages/Home";
import ExperimentDetail from "@/pages/ExperimentDetail";
import Timeline from "@/pages/Timeline";

export default function App() {
  return (
    <Router>
      <div className="bg-grid min-h-screen">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/timeline" element={<Timeline />} />
          <Route path="/experiment/:id" element={<ExperimentDetail />} />
        </Routes>
      </div>
    </Router>
  );
}
