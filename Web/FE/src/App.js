
import './App.css';
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import ObjectDetection from './pages/ObjectDetection'; // Import as default, no curly braces
import { HomePage } from './pages/HomePage';
import PredictFire from './pages/PredictFire';
import { Models } from './components/Models';
import Sentinel from './pages/Sentinel';
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/object-detection" element={<ObjectDetection />} />
        <Route path="/sentinel" element={<Sentinel />} />
        <Route path ="/predict-fire" element={<PredictFire />} />
        <Route path="/homepage" element={<HomePage />} />
      </Routes>
    </Router>
  );
}

export default App;