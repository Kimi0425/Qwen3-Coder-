#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版Python分析服务 - 用于测试上传功能
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "python-analysis"
    })

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    try:
        # 获取上传的文件
        if 'file' not in request.files:
            return jsonify({"error": "没有文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400
        
        # 获取表单数据
        name = request.form.get('name', '未命名数据')
        description = request.form.get('description', '')
        category = request.form.get('category', 'other')
        
        # 保存文件
        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # 返回成功响应
        return jsonify({
            "success": True,
            "message": "文件上传成功",
            "data": {
                "dataId": f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "name": name,
                "description": description,
                "category": category,
                "filename": filename,
                "size": os.path.getsize(filepath),
                "uploadTime": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"上传失败: {str(e)}"}), 500

@app.route('/api/data/list', methods=['GET'])
def list_data():
    try:
        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            return jsonify({"data": []})
        
        files = []
        for filename in os.listdir(upload_dir):
            filepath = os.path.join(upload_dir, filename)
            if os.path.isfile(filepath):
                files.append({
                    "dataId": f"data_{filename}",
                    "name": filename,
                    "size": os.path.getsize(filepath),
                    "uploadTime": datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                })
        
        return jsonify({"data": files})
        
    except Exception as e:
        return jsonify({"error": f"获取文件列表失败: {str(e)}"}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    return jsonify({
        "success": True,
        "message": "分析功能暂未实现",
        "results": {
            "analysis": {
                "trends": [{"column": "示例列", "direction": "上升", "strength": "中等"}],
                "insights": ["这是一个示例分析结果"]
            }
        }
    })

if __name__ == '__main__':
    print("启动简化版Python分析服务...")
    print("访问地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
