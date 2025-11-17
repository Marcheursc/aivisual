import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import AppRoutes from './routes';
import MainLayout from './layouts/MainLayout';
import './App.css';

// 导入组件样式
import './components/FileUploader.css';
import './components/DetectionControls.css';
import './components/TaskMonitor.css';
import './components/Button.css';

function App() {
  return (
    <Router>
      <MainLayout>
        <AppRoutes />
      </MainLayout>
    </Router>
  );
}

export default App;