import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from '../pages/Home';
import DetectionPage from '../pages/Detection';

const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/detection" element={<DetectionPage />} />
    </Routes>
  );
};

export default AppRoutes;
