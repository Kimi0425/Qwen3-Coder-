// 数据可视化系统前端JavaScript
class DataVisualizationApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000';
        this.currentData = null;
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDashboardData();
        this.loadDataList();
        this.setupFileUpload();
        this.setupTemperatureSlider();
    }

    setupEventListeners() {
        // 温度滑块事件
        const tempSlider = document.getElementById('model-temperature');
        if (tempSlider) {
            tempSlider.addEventListener('input', (e) => {
                document.getElementById('temp-value').textContent = e.target.value;
            });
        }

        // 预测选项事件
        const enablePrediction = document.getElementById('enable-prediction');
        if (enablePrediction) {
            enablePrediction.addEventListener('change', (e) => {
                const predictionSteps = document.getElementById('prediction-steps');
                if (e.target.checked) {
                    predictionSteps.style.display = 'block';
                } else {
                    predictionSteps.style.display = 'none';
                }
            });
        }
    }

    setupFileUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        // 文件选择
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
    }

    setupTemperatureSlider() {
        const slider = document.getElementById('model-temperature');
        if (slider) {
            slider.addEventListener('input', (e) => {
                document.getElementById('temp-value').textContent = e.target.value;
            });
        }
    }

    async handleFileSelect(file) {
        const fileInfo = document.getElementById('file-info');
        const uploadBtn = document.getElementById('upload-btn');
        
        // 显示文件信息
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-size').textContent = this.formatFileSize(file.size);
        document.getElementById('file-type').textContent = file.type || '未知';
        
        fileInfo.style.display = 'block';
        uploadBtn.disabled = false;
        
        // 预览数据
        await this.previewFileData(file);
    }

    async previewFileData(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`${this.apiBaseUrl}/api/data/preview`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                this.displayDataPreview(data);
            }
        } catch (error) {
            console.error('预览数据失败:', error);
        }
    }

    displayDataPreview(data) {
        const preview = document.getElementById('data-preview');
        const rows = data.rows || 0;
        const columns = data.columns || 0;
        
        document.getElementById('file-rows').textContent = rows;
        document.getElementById('file-columns').textContent = columns;
        
        // 显示数据预览表格
        if (data.preview && data.preview.length > 0) {
            let tableHtml = '<table class="table table-sm">';
            tableHtml += '<thead><tr>';
            Object.keys(data.preview[0]).forEach(key => {
                tableHtml += `<th>${key}</th>`;
            });
            tableHtml += '</tr></thead><tbody>';
            
            data.preview.slice(0, 5).forEach(row => {
                tableHtml += '<tr>';
                Object.values(row).forEach(value => {
                    tableHtml += `<td>${value}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table>';
            
            preview.innerHTML = tableHtml;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async uploadData() {
        const fileInput = document.getElementById('file-input');
        const dataName = document.getElementById('data-name').value;
        const dataCategory = document.getElementById('data-category').value;
        const dataDescription = document.getElementById('data-description').value;
        
        if (!fileInput.files[0]) {
            this.showAlert('请选择文件', 'warning');
            return;
        }
        
        if (!dataName) {
            this.showAlert('请输入数据名称', 'warning');
            return;
        }
        
        try {
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            formData.append('name', dataName);
            formData.append('category', dataCategory);
            formData.append('description', dataDescription);
            
            const response = await fetch(`${this.apiBaseUrl}/api/data/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.showAlert('数据上传成功', 'success');
                this.loadDataList();
                this.resetUploadForm();
            } else {
                const error = await response.json();
                this.showAlert(`上传失败: ${error.message}`, 'danger');
            }
            
        } catch (error) {
            console.error('上传失败:', error);
            this.showAlert('数据上传失败，请检查后端服务是否运行', 'danger');
        }
    }

    resetUploadForm() {
        document.getElementById('file-input').value = '';
        document.getElementById('data-name').value = '';
        document.getElementById('data-description').value = '';
        document.getElementById('file-info').style.display = 'none';
        document.getElementById('upload-btn').disabled = true;
        document.getElementById('data-preview').innerHTML = '<p class="text-muted text-center">请先选择数据文件</p>';
    }

    async loadDataList() {
        try {
            // 从后端API获取数据列表
            const response = await fetch(`${this.apiBaseUrl}/api/data/list`);
            if (response.ok) {
                const result = await response.json();
                this.updateDataSelectors(result.data || []);
            } else {
                // 如果后端接口不存在，从本地存储加载
                const data = JSON.parse(localStorage.getItem('uploadedData') || '[]');
                this.updateDataSelectors(data);
            }
        } catch (error) {
            console.error('加载数据列表失败:', error);
            // 备用方案：从本地存储加载
            const data = JSON.parse(localStorage.getItem('uploadedData') || '[]');
            this.updateDataSelectors(data);
        }
    }

    updateDataSelectors(dataList) {
        const analysisData = document.getElementById('analysis-data');
        const vizData = document.getElementById('viz-data');
        
        // 清空现有选项
        analysisData.innerHTML = '<option value="">请选择数据</option>';
        vizData.innerHTML = '<option value="">请选择数据</option>';
        
        // 添加数据选项
        dataList.forEach(item => {
            const displayName = item.name || item.fileName || '未命名数据';
            const option = `<option value="${item.dataId}">${displayName}</option>`;
            analysisData.innerHTML += option;
            vizData.innerHTML += option;
        });
    }

    async startAnalysis() {
        const dataId = document.getElementById('analysis-data').value;
        const analysisType = document.getElementById('analysis-type').value;
        const enablePrediction = document.getElementById('enable-prediction').checked;
        const predictionSteps = document.getElementById('prediction-steps-input').value;
        
        if (!dataId) {
            this.showAlert('请选择数据', 'warning');
            return;
        }
        
        const loading = document.getElementById('analysis-loading');
        const results = document.getElementById('analysis-results');
        
        loading.classList.add('show');
        results.innerHTML = '';
        
        try {
            const requestData = {
                dataId: dataId,
                analysisType: analysisType,
                enablePrediction: enablePrediction,
                predictionSteps: parseInt(predictionSteps),
                modelType: 'qwen3-coder-plus'
            };
            
            const response = await fetch(`${this.apiBaseUrl}/api/data/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (response.ok) {
                const result = await response.json();
                this.displayAnalysisResults(result);
            } else {
                throw new Error('分析失败');
            }
        } catch (error) {
            console.error('分析失败:', error);
            this.showAlert('数据分析失败', 'danger');
        } finally {
            loading.classList.remove('show');
        }
    }

    displayAnalysisResults(result) {
        const results = document.getElementById('analysis-results');
        let html = '';
        
        // 分析结果
        if (result) {
            const analysis = result;
            
            // 数据概览
            if (analysis.data_overview) {
                html += '<div class="card mb-3">';
                html += '<div class="card-header"><h6 class="mb-0">数据概览</h6></div>';
                html += '<div class="card-body">';
                html += `<p><strong>总行数:</strong> ${analysis.data_overview.total_rows}</p>`;
                html += `<p><strong>总列数:</strong> ${analysis.data_overview.total_columns}</p>`;
                html += `<p><strong>数值列:</strong> ${analysis.data_overview.numeric_columns}</p>`;
                html += `<p><strong>分类列:</strong> ${analysis.data_overview.categorical_columns}</p>`;
                html += '</div></div>';
            }
            
            // 趋势分析结果
            if (analysis.trends) {
                html += '<div class="card mb-3">';
                html += '<div class="card-header"><h6 class="mb-0">趋势分析</h6></div>';
                html += '<div class="card-body">';
                analysis.trends.forEach(trend => {
                    html += `<div class="mb-2">`;
                    html += `<strong>${trend.column}:</strong> ${trend.direction}趋势 `;
                    html += `<span class="badge bg-${trend.strength === '强' ? 'success' : 'warning'}">${trend.strength}</span>`;
                    html += `</div>`;
                });
                html += '</div></div>';
            }
            
            // 相关性分析
            if (analysis.correlations) {
                html += '<div class="card mb-3">';
                html += '<div class="card-header"><h6 class="mb-0">相关性分析</h6></div>';
                html += '<div class="card-body">';
                analysis.correlations.slice(0, 5).forEach(corr => {
                    html += `<div class="mb-2">`;
                    html += `<strong>${corr.column1} vs ${corr.column2}:</strong> `;
                    html += `<span class="badge bg-info">${corr.correlation.toFixed(3)}</span>`;
                    html += `</div>`;
                });
                html += '</div></div>';
            }
            
            // 预测结果
            if (analysis.predictions) {
                html += '<div class="card mb-3">';
                html += '<div class="card-header"><h6 class="mb-0">预测结果</h6></div>';
                html += '<div class="card-body">';
                html += `<p><strong>R²得分:</strong> ${analysis.predictions.model_performance?.r2_score?.toFixed(3) || 'N/A'}</p>`;
                html += `<p><strong>均方误差:</strong> ${analysis.predictions.model_performance?.mse?.toFixed(3) || 'N/A'}</p>`;
                html += '</div></div>';
            }
            
            // 分析结果
            if (analysis.analysis_results) {
                const results = analysis.analysis_results;
                
                // 统计摘要
                if (results.statistical_summary && Object.keys(results.statistical_summary).length > 0) {
                    html += '<div class="card mb-3">';
                    html += '<div class="card-header"><h6 class="mb-0">统计摘要</h6></div>';
                    html += '<div class="card-body">';
                    
                    Object.keys(results.statistical_summary).forEach(statType => {
                        const stats = results.statistical_summary[statType];
                        html += `<h6>${statType.toUpperCase()}</h6>`;
                        html += '<div class="row">';
                        Object.keys(stats).forEach(col => {
                            html += `<div class="col-md-6 mb-2">`;
                            html += `<strong>${col}:</strong> ${stats[col].toFixed(4)}`;
                            html += `</div>`;
                        });
                        html += '</div>';
                    });
                    
                    html += '</div></div>';
                }
                
                // 趋势分析结果
                if (results.trends && results.trends.length > 0) {
                    html += '<div class="card mb-3">';
                    html += '<div class="card-header"><h6 class="mb-0">趋势分析</h6></div>';
                    html += '<div class="card-body">';
                    results.trends.forEach(trend => {
                        html += `<div class="mb-2">`;
                        html += `<strong>${trend.column}:</strong> ${trend.direction}趋势 `;
                        html += `<span class="badge bg-${trend.strength === '强' ? 'success' : 'warning'}">${trend.strength}</span>`;
                        html += `</div>`;
                    });
                    html += '</div></div>';
                }
                
                // 相关性分析结果
                if (results.strong_correlations && results.strong_correlations.length > 0) {
                    html += '<div class="card mb-3">';
                    html += '<div class="card-header"><h6 class="mb-0">强相关性</h6></div>';
                    html += '<div class="card-body">';
                    results.strong_correlations.forEach(corr => {
                        html += `<div class="mb-2">`;
                        html += `<strong>${corr.column1} vs ${corr.column2}:</strong> `;
                        html += `<span class="badge bg-info">${corr.correlation.toFixed(3)}</span>`;
                        html += `</div>`;
                    });
                    html += '</div></div>';
                }
            }
            
            // AI洞察
            if (analysis.insights && analysis.insights.length > 0) {
                html += '<div class="card mb-3">';
                html += '<div class="card-header"><h6 class="mb-0">AI智能洞察</h6></div>';
                html += '<div class="card-body">';
                analysis.insights.forEach(insight => {
                    html += `<div class="mb-2">${insight}</div>`;
                });
                html += '</div></div>';
            }
        }
        
        results.innerHTML = html;
    }

    async generateChart() {
        const dataId = document.getElementById('viz-data').value;
        const chartType = document.getElementById('chart-type').value;
        const personSelector = document.getElementById('person-selector');
        
        if (!dataId) {
            this.showAlert('请选择数据', 'warning');
            return;
        }
        
        try {
            // 获取选中的人员
            const selectedPersons = Array.from(personSelector.selectedOptions)
                .map(option => option.value)
                .filter(value => value !== '');
            
            let url = `${this.apiBaseUrl}/api/visualize/${dataId}?chartType=${chartType}`;
            if (selectedPersons.length > 0) {
                url += `&columns=${selectedPersons.join(',')}`;
            }
            
            const response = await fetch(url);
            if (response.ok) {
                const chartData = await response.json();
                this.renderChart(chartData, chartType);
                this.updatePersonSelector(chartData.personColumns);
            } else {
                throw new Error('生成图表失败');
            }
        } catch (error) {
            console.error('生成图表失败:', error);
            this.showAlert('图表生成失败', 'danger');
        }
    }

    updatePersonSelector(personColumns) {
        const personSelector = document.getElementById('person-selector');
        if (!personSelector || !personColumns) return;
        
        // 清空现有选项（保留第一个"显示所有人员"选项）
        personSelector.innerHTML = '<option value="">显示所有人员</option>';
        
        // 添加人员选项
        personColumns.forEach(person => {
            const option = document.createElement('option');
            option.value = person;
            option.textContent = person;
            personSelector.appendChild(option);
        });
    }

    renderChart(chartData, chartType) {
        const container = document.getElementById('chart-container');
        container.innerHTML = '<div id="chart"></div>';
        
        const plotDiv = document.getElementById('chart');
        
        // 使用Plotly渲染图表
        Plotly.newPlot(plotDiv, chartData.data, chartData.layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        });
    }

    async generatePrediction(type) {
        const results = document.getElementById('prediction-results');
        results.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">正在生成预测...</p></div>';
        
        try {
            // 模拟预测结果
            setTimeout(() => {
                let html = '';
                
                if (type === 'linear') {
                    html = `
                        <div class="card">
                            <div class="card-header"><h6 class="mb-0">线性回归预测结果</h6></div>
                            <div class="card-body">
                                <p><strong>预测模型:</strong> 线性回归</p>
                                <p><strong>R²得分:</strong> 0.85</p>
                                <p><strong>预测趋势:</strong> 上升</p>
                                <p><strong>置信度:</strong> 95%</p>
                                <div class="mt-3">
                                    <canvas id="predictionChart" width="400" height="200"></canvas>
                                </div>
                            </div>
                        </div>
                    `;
                } else if (type === 'ai') {
                    html = `
                        <div class="card">
                            <div class="card-header"><h6 class="mb-0">AI智能预测结果</h6></div>
                            <div class="card-body">
                                <p><strong>预测模型:</strong> Qwen3-Coder Plus</p>
                                <p><strong>预测准确度:</strong> 92%</p>
                                <p><strong>关键发现:</strong></p>
                                <ul>
                                    <li>数据呈现明显的季节性特征</li>
                                    <li>未来3个月预计增长15-20%</li>
                                    <li>建议关注Q4季度的异常波动</li>
                                </ul>
                                <div class="mt-3">
                                    <canvas id="aiPredictionChart" width="400" height="200"></canvas>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                results.innerHTML = html;
                
                // 渲染预测图表
                this.renderPredictionChart(type);
            }, 2000);
            
        } catch (error) {
            console.error('预测失败:', error);
            this.showAlert('预测生成失败', 'danger');
        }
    }

    renderPredictionChart(type) {
        const canvasId = type === 'linear' ? 'predictionChart' : 'aiPredictionChart';
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        // 模拟数据
        const labels = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月'];
        const historicalData = [100, 120, 110, 130, 140, 150, 160, 170, 180, 190];
        const predictedData = [190, 200, 210, 220, 230, 240, 250, 260, 270, 280];
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '历史数据',
                    data: historicalData,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }, {
                    label: '预测数据',
                    data: predictedData,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderDash: [5, 5],
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: type === 'linear' ? '线性回归预测' : 'AI智能预测'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    async loadDashboardData() {
        try {
            // 加载统计数据
            const response = await fetch(`${this.apiBaseUrl}/api/dashboard/stats`);
            if (response.ok) {
                const stats = await response.json();
                this.updateDashboardStats(stats);
            }
            
            // 加载趋势图表
            this.renderTrendChart();
            
        } catch (error) {
            console.error('加载仪表板数据失败:', error);
        }
    }

    updateDashboardStats(stats) {
        document.getElementById('total-data').textContent = stats.totalData || 0;
        document.getElementById('analysis-count').textContent = stats.analysisCount || 0;
        document.getElementById('chart-count').textContent = stats.chartCount || 0;
        document.getElementById('system-status').textContent = stats.systemStatus || '运行中';
    }

    renderTrendChart() {
        const ctx = document.getElementById('trendChart').getContext('2d');
        
        // 模拟趋势数据
        const labels = ['1月', '2月', '3月', '4月', '5月', '6月'];
        const data = [65, 59, 80, 81, 56, 85];
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '数据量',
                    data: data,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '数据增长趋势'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    async refreshSystemInfo() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/system/info`);
            if (response.ok) {
                const info = await response.json();
                document.getElementById('memory-usage').textContent = info.memoryUsage + '%';
                document.getElementById('cpu-usage').textContent = info.cpuUsage + '%';
            }
        } catch (error) {
            console.error('刷新系统信息失败:', error);
        }
    }

    showAlert(message, type = 'info') {
        // 创建提示框
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // 自动移除
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }
}

// 全局函数
function showSection(sectionName) {
    // 隐藏所有section
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
    
    // 显示选中的section
    const targetSection = document.getElementById(sectionName + '-section');
    if (targetSection) {
        targetSection.style.display = 'block';
    }
    
    // 更新导航状态
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.classList.remove('active');
    });
    
    event.target.classList.add('active');
}

function uploadData() {
    app.uploadData();
}

function startAnalysis() {
    app.startAnalysis();
}

function generateChart() {
    app.generateChart();
}

function generatePrediction(type) {
    app.generatePrediction(type);
}

function refreshSystemInfo() {
    app.refreshSystemInfo();
}

// 初始化应用
const app = new DataVisualizationApp();
