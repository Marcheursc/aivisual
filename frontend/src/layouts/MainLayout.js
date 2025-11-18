import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './MainLayout.css';

const MainLayout = ({ children }) => {
  const location = useLocation();

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">AI Visual Detection System</h1>
          <nav className="main-navigation">
            <ul>
              <li>
                <Link
                  to="/"
                  className={location.pathname === '/' ? 'active' : ''}
                >
                  首页
                </Link>
              </li>
              <li>
                <Link
                  to="/detection"
                  className={location.pathname === '/detection' ? 'active' : ''}
                >
                  视频检测
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main className="app-main">
        {children}
      </main>

      <footer className="app-footer">
        <p>&copy; 2025 AI Visual Detection System. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default MainLayout;
