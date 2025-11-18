import React, { useState } from 'react';
import { FileUploader, DetectionControls, TaskMonitor, Button } from '../../components';
import './Detection.css';

const DetectionPage = () => {
  // 文件上传状态
  const [fileId, setFileId] = useState('');
  const [taskId, setTaskId] = useState('');

  // 处理文件上传完成事件
  const handleFileUploaded = (fileId) => {
    setFileId(fileId);
  };

  // 处理开始检测事件
  const handleStartDetection = async (params) => {
    if (!fileId) {
      alert('请先上传文件并获取文件ID');
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/process_video/?file_id=${fileId}&${new URLSearchParams(params)}`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setTaskId(data.task_id);
    } catch (error) {
      alert('处理启动失败: ' + error.message);
    }
  };

  // 定义检测类型及其参数
  const detectionTypes = [
    {
      value: 'loitering',
      label: '徘徊检测',
      parameters: [
        {
          name: 'loitering_time_threshold',
          label: '徘徊时间阈值',
          type: 'number',
          default: 20,
          min: 1,
          unit: '秒'
        }
      ]
    },
    {
      value: 'gather',
      label: '聚集检测',
      parameters: [
        {
          name: 'gather_threshold',
          label: '聚集人数阈值',
          type: 'number',
          default: 5,
          min: 1,
          unit: '人'
        }
      ]
    },
    {
      value: 'leave',
      label: '离岗检测',
      parameters: [
        {
          name: 'leave_threshold',
          label: '离岗时间阈值',
          type: 'number',
          default: 5,
          min: 1,
          unit: '秒'
        }
      ]
    }
  ];

  return (
    <div className="detection-page">
      <h1>视频检测</h1>

      {/* 视频上传部分 */}
      <section className="upload-section">
        <h2>视频上传</h2>
        <FileUploader
          onFileUploaded={handleFileUploaded}
          uploadEndpoint="http://localhost:8000/upload/"
          allowedTypes="video/*"
        />
      </section>

      {/* 检测控制部分 */}
      <section className="detection-section">
        <h2>检测控制</h2>
        <DetectionControls
          onStartDetection={handleStartDetection}
          detectionTypes={detectionTypes}
        />
      </section>

      {/* 任务状态部分 */}
      <section className="status-section">
        <h2>任务状态</h2>
        <TaskMonitor
          taskId={taskId}
          statusEndpoint="http://localhost:8000/task_status"
          downloadEndpoint="http://localhost:8000/download_processed"
        />
      </section>

    </div>
  );
};

export default DetectionPage;
