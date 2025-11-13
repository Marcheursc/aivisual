from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import requests
import os
import json

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your-secret-key-here'  # 在生产环境中应该使用更安全的密钥

# API服务器地址
API_BASE_URL = "http://localhost:8000"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('没有选择文件')
        return redirect(request.url)
    
    if file:
        # 上传到API服务器
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(f"{API_BASE_URL}/upload/", files=files)
        
        if response.status_code == 200:
            result = response.json()
            flash(f'文件上传成功！文件ID: {result["file_id"]}')
            return redirect(url_for('detection'))
        else:
            flash('文件上传失败')
            return redirect(request.url)

@app.route('/api/simple_detect', methods=['POST'])
def api_simple_detect():
    if 'video' not in request.files:
        return jsonify({'error': '没有上传视频文件'}), 400
    
    video = request.files['video']
    detection_type = request.form.get('detectionType', 'loitering')
    
    if video.filename == '':
        return jsonify({'error': '没有选择视频文件'}), 400
    
    # 先上传文件
    files = {'file': (video.filename, video.stream, video.content_type)}
    upload_response = requests.post(f"{API_BASE_URL}/upload/", files=files)
    
    if upload_response.status_code != 200:
        return jsonify({'error': '文件上传失败'}), 500
    
    upload_result = upload_response.json()
    file_id = upload_result['file_id']
    
    # 根据检测类型调用不同的API
    if detection_type == 'loitering':
        # 徘徊检测
        loitering_time_threshold = request.form.get('loitering_time_threshold', 20)
        params = {
            'file_id': file_id,
            'detect_loitering': True,
            'loitering_time_threshold': loitering_time_threshold,
            'detection_type': 'loitering'
        }
        response = requests.post(f"{API_BASE_URL}/process_video/", params=params)
    elif detection_type == 'gather':
        # 聚集检测
        params = {
            'file_id': file_id,
            'detection_type': 'gather'
        }
        response = requests.post(f"{API_BASE_URL}/process_video/", params=params)
    elif detection_type == 'leave':
        # 离岗检测
        params = {
            'file_id': file_id,
            'detection_type': 'leave'
        }
        response = requests.post(f"{API_BASE_URL}/process_video/", params=params)
    
    if response.status_code == 200:
        result = response.json()
        return jsonify(result)
    else:
        return jsonify({'error': '检测启动失败'}), 500

@app.route('/api/task_status/<task_id>')
def api_task_status(task_id):
    response = requests.get(f"{API_BASE_URL}/task_status/{task_id}")
    
    if response.status_code == 200:
        result = response.json()
        return jsonify(result)
    else:
        return jsonify({'error': '获取任务状态失败'}), 500

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)