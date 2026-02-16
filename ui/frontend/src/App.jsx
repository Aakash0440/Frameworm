import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Experiments from './pages/Experiments';
import Models from './pages/Models';
import Training from './pages/Training';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <nav className="sidebar">
          <h1>FRAMEWORM</h1>
          <ul>
            <li><Link to="/">Dashboard</Link></li>
            <li><Link to="/experiments">Experiments</Link></li>
            <li><Link to="/models">Models</Link></li>
            <li><Link to="/training">Training</Link></li>
          </ul>
        </nav>
        
        <main className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/models" element={<Models />} />
            <Route path="/training" element={<Training />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;