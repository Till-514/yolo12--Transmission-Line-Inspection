import requests
from datetime import datetime
import json
import sqlite3
from openai import OpenAI

from PySide6.QtWidgets import QPushButton, QTextEdit, QDialog, QVBoxLayout
from PySide6.QtCore import QThread, Signal


class AIAnalysisThread(QThread):
    analysis_complete = Signal(dict)  # 分析完成信号
    analysis_error = Signal(str)      # 错误信号
    
    def __init__(self, api_key, base_url):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.detection_data = None
        
    def set_data(self, detection_data):
        self.detection_data = detection_data
        
    def run(self):
        try:
            if not self.api_key:
                raise Exception("未设置API密钥")
                
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            # 构建提示词
            prompt = f"""
            请分析以下输电部件检测数据，并提供专业的分析报告：

            检测数据：
            {self.detection_data}

            请从以下几个方面进行分析：
            1. 不同部件正常或者破损评估
            2. 风险等级判断
            3. 建议措施
            4. 趋势预测

            请用中文回答，并保持专业性和可读性。
            """
            
            # 调用API
            response = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "system", "content": "你是一个专业的电力高级工程师资质（8年以上经验）的输电线路部件分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            # 解析响应
            ai_message = response.choices[0].message.content
            result = {
                'ai_analysis': ai_message,
                'timestamp': datetime.now()
            }
            
            self.analysis_complete.emit(result)
            
        except Exception as e:
            self.analysis_error.emit(str(e))

class AIFatigueAnalyzer:
    def __init__(self):
        self.api_key = ""  # 替换为您的Deepseek API密钥
        self.base_url = "https://api.deepseek.cn/v1"
        self.analysis_thread = None
        
    def analyze_detection_data(self, detection_data):
        """使用Deepseek分析检测数据（异步）"""
        if not self.analysis_thread:
            self.analysis_thread = AIAnalysisThread(self.api_key, self.base_url)
            
        self.analysis_thread.set_data(detection_data)
        self.analysis_thread.start()
        return self.analysis_thread

class AnalysisDialog(QDialog):
    def __init__(self, analysis_result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输电部件检测分析报告")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # 创建文本显示区域
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(analysis_result['ai_analysis'])
        
        layout.addWidget(text_edit)
        
        # 添加关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        # 设置样式
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #BDC3C7;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """) 