import React from 'react';
import './Home.css';

const Home = () => {
  return (
    <div className="home-page">
      <div className="hero-section">
        <h1>欢迎使用AI视觉检测系统</h1>
        <p>基于先进的计算机视觉技术，实时监控和分析视频内容</p>
      </div>

      <div className="features-section">
        <div className="feature-card">
          <h3>徘徊检测</h3>
          <p>检测人员在特定区域长时间逗留的行为</p>
        </div>

        <div className="feature-card">
          <h3>聚集检测</h3>
          <p>识别区域内人群聚集的情况</p>
        </div>

        <div className="feature-card">
          <h3>离岗检测</h3>
          <p>监控关键岗位人员是否离开工作岗位</p>
        </div>
      </div>

      <div className="cta-section">
        <p>点击导航栏中的"视频检测"开始使用系统</p>
      </div>
    </div>
  );
};

export default Home;
