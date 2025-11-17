import React, { useState } from 'react';
import './FileUploader.css';

const FileUploader = ({ onFileUploaded, uploadEndpoint, allowedTypes = '*' }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [fileId, setFileId] = useState('');

  // 处理文件选择
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  // 处理文件上传
  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('请选择一个文件');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setUploadStatus('上传中...');
      const response = await fetch(uploadEndpoint, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setFileId(data.file_id);
      setUploadStatus(`上传成功! 文件ID: ${data.file_id}`);
      onFileUploaded && onFileUploaded(data.file_id, data);
    } catch (error) {
      setUploadStatus('上传失败: ' + error.message);
    }
  };

  return (
    <div className="file-uploader">
      <div className="upload-controls">
        <input 
          type="file" 
          accept={allowedTypes} 
          onChange={handleFileChange} 
        />
        <button onClick={handleUpload}>上传文件</button>
        <p className="status">{uploadStatus}</p>
        {fileId && <p className="file-id">文件ID: {fileId}</p>}
      </div>
    </div>
  );
};

export default FileUploader;