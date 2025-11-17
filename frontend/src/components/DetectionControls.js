import React, { useState } from 'react';
import './DetectionControls.css';

const DetectionControls = ({ onStartDetection, detectionTypes, initialParams = {} }) => {
  const [detectionType, setDetectionType] = useState(detectionTypes[0]?.value || '');
  const [params, setParams] = useState(initialParams);

  // 处理参数变化
  const handleParamChange = (paramName, value) => {
    setParams(prev => ({
      ...prev,
      [paramName]: value
    }));
  };

  // 开始检测
  const handleStart = () => {
    onStartDetection && onStartDetection({
      detection_type: detectionType,
      ...params
    });
  };

  return (
    <div className="detection-controls">
      <div className="form-group">
        <label>
          检测类型:
          <select 
            value={detectionType} 
            onChange={(e) => setDetectionType(e.target.value)}
          >
            {detectionTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      
      {/* 渲染动态参数输入框 */}
      {detectionTypes
        .find(type => type.value === detectionType)
        ?.parameters.map(param => (
          <div className="form-group" key={param.name}>
            <label>
              {param.label}:
              <input 
                type={param.type || "number"}
                value={params[param.name] || param.default || ''}
                onChange={(e) => handleParamChange(param.name, e.target.value)}
                min={param.min}
                max={param.max}
                step={param.step}
              />
              {param.unit && <span className="unit">{param.unit}</span>}
            </label>
          </div>
        ))
      }
      
      <button onClick={handleStart}>开始检测</button>
    </div>
  );
};

export default DetectionControls;