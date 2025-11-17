import React, { useState } from 'react';
import './TaskMonitor.css';

const TaskMonitor = ({ taskId, statusEndpoint, downloadEndpoint }) => {
  const [taskStatus, setTaskStatus] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  // 检查任务状态
  const checkStatus = async () => {
    if (!taskId) {
      setStatusMessage('请先启动任务获取任务ID');
      return;
    }

    try {
      const response = await fetch(`${statusEndpoint}/${taskId}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setTaskStatus(data);
      setStatusMessage('');
    } catch (error) {
      setStatusMessage('获取任务状态失败: ' + error.message);
      setTaskStatus(null);
    }
  };

  // 下载处理后的文件
  const handleDownload = async () => {
    if (!taskId) {
      setStatusMessage('请先启动任务获取任务ID');
      return;
    }

    try {
      const response = await fetch(`${downloadEndpoint}/${taskId}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const blob = await response.blob();
      
      // 创建下载链接
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `processed_file_${taskId}.mp4`);
      document.body.appendChild(link);
      link.click();
      
      // 清理
      link.parentNode.removeChild(link);
    } catch (error) {
      setStatusMessage('下载失败: ' + error.message);
    }
  };

  return (
    <div className="task-monitor">
      <div className="monitor-controls">
        <button onClick={checkStatus}>检查状态</button>
        <button 
          onClick={handleDownload} 
          disabled={!taskStatus || taskStatus.status !== 'completed'}
        >
          下载处理后的文件
        </button>
        <p className="status">{statusMessage}</p>
        
        {taskStatus && (
          <div className="task-details">
            <h3>状态详情</h3>
            <p>状态: {taskStatus.status}</p>
            {taskStatus.progress !== undefined && <p>进度: {taskStatus.progress}%</p>}
            {taskStatus.frame_count && <p>处理帧数: {taskStatus.frame_count}</p>}
          </div>
        )}
      </div>
    </div>
  );
};

export default TaskMonitor;