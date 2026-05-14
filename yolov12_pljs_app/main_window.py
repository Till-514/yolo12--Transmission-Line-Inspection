import smtplib
import sys
import json
import os
import sqlite3
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFileDialog,
                               QFrame, QGroupBox, QLineEdit, QInputDialog, QMessageBox,
                               QDialog, QFormLayout, QTableWidget, QTableWidgetItem,
                               QHeaderView, QTextEdit, QScrollArea, QDialog, QGridLayout)
from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import time
import cv2
import numpy as np

def get_cls_color(cls_names):
    colors = {}
    np.random.seed(42)
    for i, name in enumerate(cls_names):
        colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
    return colors

def drawRectBox(image, bbox, label=None, color=(0, 255, 0), thickness=2, alpha=0.2, addText=True):
    x1, y1, x2, y2 = map(int, bbox)
    # 画框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    # 画标签
    if label and addText:
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

def cv_imread(file_path):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
from QtFusion.path import abs_path
from QtFusion.config import QF_Config
from YOLOv12Model import YOLOv12Detector
from ai_analyzer import AIFatigueAnalyzer, AnalysisDialog
from voice_assistant import VoiceAssistant
from mail_sender import send_analysis_report
import shutil
from pathlib import Path
from PIL import Image
from qr_share import QRShareDialog


def initialize_database(db_path='data_store.db'):
    """初始化数据库，添加必要的列（如果不存在）"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 检查processed列是否已存在
        cursor.execute("PRAGMA table_info(detection_results)")
        columns = [info[1] for info in cursor.fetchall()]

        if 'processed' not in columns:
            # 添加processed列
            cursor.execute('''
                           ALTER TABLE detection_results
                               ADD COLUMN processed INTEGER DEFAULT 0
                           ''')
            conn.commit()
            print("成功添加processed列到detection_results表")
        else:
            print("processed列已存在，无需添加")

        conn.close()
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")


# 在应用启动时调用
if __name__ == "__main__":
    # 先初始化数据库
    initialize_database()

QF_Config.set_verbose(False)

# 定义固定的显示尺寸
DISPLAY_WIDTH = 1180
DISPLAY_HEIGHT = 600
# 定义左侧面板宽度
LEFT_PANEL_WIDTH = 250
# 默认模型路径
DEFAULT_MODEL_PATH = "weights/weights/best_high_1.pt"
# 配置文件路径
CONFIG_FILE = "config.json"
# 数据库路径
DB_FILE = "data_store.db"
# 检测图片存档路径
ARCHIVE_ROOT = Path("archive")  # 一键存档根目录
ARCHIVE_ROOT.mkdir(exist_ok=True)


class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_model_path=None, db_config=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(400)
        self.setStyleSheet("""
            font-size: 14px;
            QDialog {
                background-color: #121212;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #424242;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0288d1;
                color: #ffffff;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffffff;
            }
            QLineEdit {
                background-color: #2c2c2c;
                color: #e0e0e0;
                border: 1px solid #424242;
                border-radius: 4px;
                padding: 4px;
            }
        """)

        # 创建布局
        layout = QVBoxLayout(self)

        # 模型选择组
        model_group = QGroupBox("权重模型设置")
        model_layout = QVBoxLayout()

        # 当前模型路径显示
        self.model_path_label = QLabel(current_model_path or "未选择权重模型")
        self.model_path_label.setWordWrap(True)

        # 选择模型按钮
        self.select_model_btn = QPushButton("选择权重模型文件")
        self.select_model_btn.clicked.connect(self.select_model)

        model_layout.addWidget(QLabel("当前模型:"))
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.select_model_btn)
        model_group.setLayout(model_layout)

        # 数据库设置组
        db_group = QGroupBox("数据库设置")
        db_layout = QFormLayout()

        # 数据库路径输入
        self.db_path_input = QLineEdit(db_config.get('db_path', DB_FILE) if db_config else DB_FILE)
        db_layout.addRow("数据库路径:", self.db_path_input)

        # 选择数据库按钮
        self.select_db_btn = QPushButton("选择数据库文件")
        self.select_db_btn.clicked.connect(self.select_database)
        db_layout.addRow("", self.select_db_btn)

        db_group.setLayout(db_layout)

        # 确定和取消按钮
        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("确定")
        self.cancel_button = QPushButton("取消")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)

        layout.addWidget(model_group)
        layout.addWidget(db_group)
        layout.addLayout(buttons_layout)

        self.selected_model_path = current_model_path

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择权重模型文件",
            "",
            "模型文件 (*.pt)"
        )
        if file_name:
            self.model_path_label.setText(file_name)
            self.selected_model_path = file_name

    def select_database(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "选择数据库文件",
            "",
            "数据库文件 (*.db)"
        )
        if file_name:
            self.db_path_input.setText(file_name)

    def get_selected_model_path(self):
        return self.selected_model_path

    def get_db_config(self):
        return {
            'db_path': self.db_path_input.text()
        }


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
        # 用于记录每种检测类型的最后保存时间
        self.last_save_time = {}

    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建检测结果表
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS detection_results
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           timestamp
                           DATETIME
                           NOT
                           NULL,
                           detection_type
                           TEXT
                           NOT
                           NULL,
                           confidence
                           REAL
                           NOT
                           NULL
                       )
                       ''')

        conn.commit()
        conn.close()

    def save_detection(self, detection_type, confidence):
        """保存检测结果，置信度保留到小数点后三位"""
        # 将置信度保留到小数点后三位
        rounded_confidence = round(confidence, 3)

        # 获取当前时间并格式化为只保留到秒
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # 再次检查是否已经在同一秒内保存过检测结果
        current_second = int(time.time())
        for name, save_time in self.last_save_time.items():
            if int(save_time) == current_second:
                print(f"保存检查：同一秒内已保存过 {name}，跳过保存 {detection_type}")
                return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO detection_results (timestamp, detection_type, confidence)
                       VALUES (?, ?, ?)
                       ''', (formatted_time, detection_type, rounded_confidence))

        conn.commit()
        conn.close()

        # 更新最后保存时间
        self.last_save_time[detection_type] = time.time()

        print(f"已保存检测结果: {detection_type}, 置信度: {rounded_confidence:.3f}, 时间: {formatted_time}")


class DetectionRecordDialog(QDialog):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setWindowTitle("检测记录")
        self.setMinimumSize(600, 400)
        self.setStyleSheet("""
            font-size: 14px;
            QDialog {
                background-color: #121212;
                color: #ffffff;
            }
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #2c2c2c;
                color: #ffffff;
                gridline-color: #424242;
                border: 1px solid #424242;
                font-size: 14px;
                font-weight: bold;
            }
            QHeaderView::section {
                background-color: #0288d1;
                color: #ffffff;
                padding: 4px;
                border: none;
            }
            QPushButton {
                background-color: #0288d1;
                color: #ffffff;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #039be5;
            }
        """)

        layout = QVBoxLayout(self)

        # 创建表格
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["时间", "检测类型", "置信度"])

        # 设置表格样式
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
            }
            QHeaderView::section {
                background-color: #2C3E50;
                color: white;
                padding: 5px;
                border: none;
            }
        """)

        # 设置表格列宽
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        layout.addWidget(self.table)

        # 添加刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.load_records)
        layout.addWidget(refresh_btn)

        # 加载记录
        self.load_records()

    def load_records(self):
        """加载检测记录"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()

            # 获取最近的100条记录
            cursor.execute('''
                           SELECT timestamp, detection_type, confidence
                           FROM detection_results
                           ORDER BY timestamp DESC
                               LIMIT 100
                           ''')

            records = cursor.fetchall()
            conn.close()

            # 更新表格
            self.table.setRowCount(len(records))
            for i, record in enumerate(records):
                timestamp = record[0]
                # 处理可能带有毫秒的旧记录格式
                if '.' in timestamp:
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = timestamp  # 直接使用新的格式化时间

                self.table.setItem(i, 0, QTableWidgetItem(time_str))
                self.table.setItem(i, 1, QTableWidgetItem(record[1]))
                self.table.setItem(i, 2, QTableWidgetItem(f"{record[2]:.3f}"))

                # 设置交替行颜色
                if i % 2 == 0:
                    for j in range(3):
                        self.table.item(i, j).setBackground(Qt.lightGray)

        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载记录失败: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("天轨智寻")
        self.setMinimumSize(1200, 800)

        # 加载配置
        self.config = self.load_config()
        self.current_model_path = self.config.get('model_path', DEFAULT_MODEL_PATH)
        self.db_config = self.config.get('db_config', {'db_path': DB_FILE})

        # 初始化数据库管理器
        self.db_manager = DatabaseManager(self.db_config['db_path'])

        # 初始化AI分析器
        self.ai_analyzer = AIFatigueAnalyzer()

        if 'ai_api_key' in self.config:
            self.ai_analyzer.set_api_key(self.config['ai_api_key'])

        # 初始化语音助手
        self.voice_assistant = VoiceAssistant(self.db_config['db_path'])

        # 初始化AI分析报告内容
        self.latest_ai_report = None

        # 初始化模型
        self.model = YOLOv12Detector()
        self.load_model(self.current_model_path)
        self.colors = get_cls_color(self.model.names)

        # 初始化视频处理器和按钮状态
        self.video_handler = None
        self.aianalyzer_start_btn = None
        self.aianalyzer_stop_btn = None
        self.ip_input = None

        # 初始化检测结果保存控制
        # self.last_save_time = {}  # 已经移动到DatabaseManager中
        self.save_interval = 1  # 设置保存间隔为1秒
        self.confidence_threshold = 0.6  # 设置置信度阈值
        self.last_detection = {}  # 记录上一次的检测结果
        self.consecutive_count = {}  # 记录连续检测次数

        # 创建主窗口部件和布局
        main_widget = QWidget()
        main_widget.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #e0e0e0;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 14px;
            }
        """)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建左侧功能面板
        left_panel = self.create_left_panel()
        left_panel.setFixedWidth(LEFT_PANEL_WIDTH)
        main_layout.addWidget(left_panel)

        # 创建右侧显示区域
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel)

        # 创建设置按钮和检测记录按钮
        self.create_settings_button()

        main_layout.setSpacing(0)

        if 'smtp_config' not in self.config:
            self.config['smtp_config'] = {
                "server": "smtp.qq.com",
                "port": 465,
                "user": "2040043330@qq.com",
                "password": "vjzqymwvxkyvbejh"
            }
        self.showMaximized()

    def create_settings_button(self):
        """创建设置按钮和检测记录按钮"""
        # 创建按钮容器
        button_container = QWidget(self)
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)

        # 创建设置按钮
        self.settings_btn = QPushButton("设置")
        btn_style = """
            QPushButton {
                background-color: #0288d1;
                color: #ffffff;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #039be5;
            }
        """
        self.settings_btn.setStyleSheet(btn_style)
        # self.record_btn.setStyleSheet(btn_style)
        # self.ai_analysis_btn.setStyleSheet(btn_style)
        self.settings_btn.clicked.connect(self.show_settings)

        # 创建检测记录按钮
        self.record_btn = QPushButton("记录")
        btn_style = """
            QPushButton {
                background-color: #0288d1;
                color: #ffffff;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #039be5;
            }
        """
        # self.settings_btn.setStyleSheet(btn_style)
        self.record_btn.setStyleSheet(btn_style)
        # self.ai_analysis_btn.setStyleSheet(btn_style)
        self.record_btn.clicked.connect(self.show_records)

        # 创建AI分析按钮
        self.ai_analysis_btn = QPushButton("KiMi分析")
        self.ai_analysis_btn.clicked.connect(self.show_ai_analysis)

        # 设置按钮大小
        self.settings_btn.setFixedSize(60, 30)
        self.record_btn.setFixedSize(60, 30)

        # 添加按钮到布局
        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.settings_btn)

        # 将按钮容器添加到窗口右上角
        button_container.setFixedSize(220, 30)
        button_container.move(self.width() - 230, 10)

    def resizeEvent(self, event):
        """窗口大小改变时重新定位按钮"""
        super().resizeEvent(event)
        if hasattr(self, 'settings_btn'):
            button_container = self.settings_btn.parent()
            button_container.move(self.width() - 230, 10)

    def show_settings(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self, self.current_model_path, self.db_config)
        if dialog.exec() == QDialog.Accepted:
            new_model_path = dialog.get_selected_model_path()
            new_db_config = dialog.get_db_config()

            if new_model_path and new_model_path != self.current_model_path:
                try:
                    self.load_model(new_model_path)
                    self.current_model_path = new_model_path
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"加载模型失败: {str(e)}")
                    return

            if new_db_config['db_path'] != self.db_config['db_path']:
                self.db_config = new_db_config
                self.db_manager = DatabaseManager(self.db_config['db_path'])

            self.save_config()
            QMessageBox.information(self, "成功", "设置已更新")

    def load_model(self, model_path):
        """加载模型"""
        try:
            self.model.load_model(abs_path(model_path, path_type="current"))
            self.colors = get_cls_color(self.model.names)
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")

    def load_config(self):
        """加载配置"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    # 如果存在API密钥，立即设置
                    if 'ai_api_key' in config:
                        self.ai_analyzer.set_api_key(config['ai_api_key'])
                    return config
            except:
                return {}
        return {}

    def save_config(self):
        """保存配置"""
        config = {
            'model_path': self.current_model_path,
            'db_config': self.db_config,
            'ai_api_key': self.ai_analyzer.api_key,
            'smtp_config': self.config.get('smtp_config', {
                "server": "smtp.qq.com",
                "port": 465,
                "user": "2040043330@qq.com",
                "password": "vjzqymwvxkyvbejh"
            })
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"保存配置失败: {str(e)}")

    def create_left_panel(self):
        """创建左侧功能面板"""
        left_panel = QWidget()
        left_panel.setStyleSheet("""
            QWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
        }
        QPushButton {
            background-color: #323232;
            border: 1px solid #424242;
            padding: 8px;
            border-radius: 6px;
            margin: 4px 8px;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #424242;
            border-color: #0288d1;
        }
        QPushButton:pressed {
            background-color: #0288d1;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #424242;
            margin-top: 10px;
            margin-left: 6px;
            margin-right: 6px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #ffffff;
            margin-left: 4px;
            color: #e0e0e0;
        }
        QLabel {
            margin: 4px 8px;
            color: #bdbdbd;
        }
        /* 追加到现有样式表末尾 */
        QLabel {
            font-size: 12px;          /* 默认14→12 */
        }
        QPushButton {
            font-size: 12px;
            padding: 3px;             /* 减少上下留白 */
            min-height: 24px;         /* 默认30→24 */
        }
        QGroupBox {
            margin-top: 6px;          /* 减少组间距 */
            padding-top: 10px;
        }
    """)

        layout = QVBoxLayout(left_panel)
        layout.setContentsMargins(0, 10, 0, 10)  # 设置左侧面板的内边距

        # 功能选择标题
        title_label = QLabel("功能选择")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        layout.addWidget(title_label)

        # 图片检测组
        image_group = QGroupBox("图片检测")
        image_layout = QVBoxLayout(image_group)
        self.image_path_label = QLabel("选择文件")
        self.image_select_btn = QPushButton("选择文件")
        self.image_start_btn = QPushButton("开始检测")
        self.image_stop_btn = QPushButton("关闭检测")

        image_layout.addWidget(self.image_path_label)
        image_layout.addWidget(self.image_select_btn)
        image_layout.addWidget(self.image_start_btn)
        image_layout.addWidget(self.image_stop_btn)
        layout.addWidget(image_group)

        # 视频文件检测组
        video_group = QGroupBox("视频文件检测")
        video_layout = QVBoxLayout(video_group)
        self.video_path_label = QLabel("选择文件")
        self.video_select_btn = QPushButton("选择文件")
        self.video_start_btn = QPushButton("开始检测")
        self.video_stop_btn = QPushButton("关闭检测")

        video_layout.addWidget(self.video_path_label)
        video_layout.addWidget(self.video_select_btn)
        video_layout.addWidget(self.video_start_btn)
        video_layout.addWidget(self.video_stop_btn)
        layout.addWidget(video_group)

        # 实时视频检测组
        realtime_group = QGroupBox("实时视频检测")
        realtime_layout = QVBoxLayout(realtime_group)
        self.camera_start_btn = QPushButton("开启摄像头")
        self.camera_stop_btn = QPushButton("关闭摄像头")

        realtime_layout.addWidget(self.camera_start_btn)
        realtime_layout.addWidget(self.camera_stop_btn)
        layout.addWidget(realtime_group)

        # AI分析组
        aianalyzer_group = QGroupBox("KiMi分析检测")    
        aianalyzer_layout = QVBoxLayout(aianalyzer_group)

        # 添加AI分析控制按钮
        self.aianalyzer_start_btn = QPushButton("进行KiMi分析")
        self.aianalyzer_stop_btn = QPushButton("发送KiMi分析报告")

        aianalyzer_layout.addWidget(self.aianalyzer_start_btn)
        aianalyzer_layout.addWidget(self.aianalyzer_stop_btn)
        layout.addWidget(aianalyzer_group)

        # 语音助手控制组
        voice_group = QGroupBox("语音助手") 
        voice_layout = QVBoxLayout(voice_group)

        # 启动语音助手按钮
        self.start_voice_btn = QPushButton("启动千巡")
        self.start_voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
        """)
        self.start_voice_btn.clicked.connect(self.start_voice_assistant)

        # 停止语音助手按钮
        self.stop_voice_btn = QPushButton("停止千巡")
        self.stop_voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        self.stop_voice_btn.clicked.connect(self.stop_voice_assistant)
        self.stop_voice_btn.setEnabled(False)

        voice_layout.addWidget(self.start_voice_btn)
        voice_layout.addWidget(self.stop_voice_btn)
        layout.addWidget(voice_group)

        # 添加底部弹性空间
        layout.addStretch()

        # 连接信号和槽
        self.setup_connections()

        return left_panel

    def create_right_panel(self):
        """创建右侧显示区域"""
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #e0e0e0;
            }
        """)
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(10, 10, 10, 10)  # 设置右侧面板的内边距

        # 显示标题
        title = QLabel("天轨智寻——基于YOLOv12和KiMi分析的输电线路部件检测系统")
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: #00e5ff;
            padding: 20px;
            background-color: transparent;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)  # 设置固定大小
        self.image_label.setStyleSheet("""
            border: 2px solid #0288d1;
            background-color: #181818;
            border-radius: 8px;
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)  # 关闭拉伸（修复核心！）

        # 创建一个容器来居中显示图像标签
        image_container = QWidget()
        image_container_layout = QHBoxLayout(image_container)
        image_container_layout.addWidget(self.image_label)
        image_container_layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_container)

        # 检测结果表格
        results_group = QGroupBox("检测结果")
        results_group.setStyleSheet("""
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #424242;
                margin-top: 10px;
                padding: 10px;
                color: #00e5ff;
            }
        """)
        results_layout = QVBoxLayout(results_group)

        # 创建结果显示标签
        self.result_label = QLabel("检测结果将在这里显示")
        self.result_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.result_label)

        layout.addWidget(results_group)

        # 1. 在 results_layout 里添加“保存”按钮
        self.save_btn = QPushButton("一键存档")
        self.save_btn.setFixedHeight(32)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #323232;
                border: 1px solid #424242;
                padding: 8px;
                border-radius: 6px;
                margin: 4px 8px;
                font-size: 12px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #424242;
                border-color: #0288d1;
            }
            QPushButton:pressed {
                background-color: #0288d1;
                color: #ffffff;
            }
        """)
        self.save_btn.clicked.connect(self.save_defect)
        results_layout.addWidget(self.save_btn)

        # 2. 图库入口按钮
        self.gallery_btn = QPushButton("本地图库")
        self.gallery_btn.setFixedHeight(32)
        self.gallery_btn.setStyleSheet("""
            QPushButton {
                background-color: #323232;
                border: 1px solid #424242;
                padding: 8px;
                border-radius: 6px;
                margin: 4px 8px;
                font-size: 12px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #424242;
                border-color: #0288d1;
            }
            QPushButton:pressed {
                background-color: #0288d1;
                color: #ffffff;
            }
        """)
        self.gallery_btn.clicked.connect(self.open_gallery)
        results_layout.addWidget(self.gallery_btn)

        return right_panel

    def setup_connections(self):
        """设置信号和槽的连接"""
        # 图片检测
        self.image_select_btn.clicked.connect(self.select_image)
        self.image_start_btn.clicked.connect(self.start_image_detection)
        self.image_stop_btn.clicked.connect(self.stop_image_detection)

        # 视频检测
        self.video_select_btn.clicked.connect(self.select_video)
        self.video_start_btn.clicked.connect(self.start_video_detection)
        self.video_stop_btn.clicked.connect(self.stop_video_detection)

        # 实时视频检测
        self.camera_start_btn.clicked.connect(self.start_camera)
        self.camera_stop_btn.clicked.connect(self.stop_camera)

        # AI分析检测
        self.aianalyzer_start_btn.clicked.connect(self.start_aianalyzer)
        self.aianalyzer_stop_btn.clicked.connect(self.send_aianalyzer)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.image_path_label.setText(file_name)
            # 显示选择的图片
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "视频文件 (*.mp4 *.avi *.mov)"
        )
        if file_name:
            self.video_path_label.setText(file_name)

    def should_save_detection(self, name, conf):
        """判断是否应该保存检测结果"""
        current_time = time.time()
        current_second = int(current_time)  # 获取当前的秒时间戳

        # 检查时间间隔 - 确保同一种检测类型不会在同一秒内多次保存
        if name in self.db_manager.last_save_time and \
                current_second == int(self.db_manager.last_save_time[name]):  # 比较秒级时间戳
            print(f"同一秒内已保存过 {name}，跳过")
            return False

        # 检查置信度阈值
        if conf < self.confidence_threshold:
            return False

        # 检查连续检测结果
        if name in self.last_detection:
            last_conf = self.last_detection[name]
            # 如果置信度变化太大，可能是误检
            if abs(conf - last_conf) > 0.3:
                self.consecutive_count[name] = 0
                return False

            # 增加连续检测计数
            self.consecutive_count[name] = self.consecutive_count.get(name, 0) + 1
            # 至少需要连续2次检测才保存
            if self.consecutive_count[name] < 2:
                return False
        else:
            # 第一次检测到该类型
            self.consecutive_count[name] = 1
            self.last_detection[name] = conf
            return False

        # 防止同一秒内保存多个不同的检测状态
        # 检查是否有其他类型的检测在当前秒内被保存
        for other_name, save_time in self.db_manager.last_save_time.items():
            if other_name != name and int(save_time) == current_second:
                print(f"同一秒内已保存过其他状态 {other_name}，跳过保存 {name}")
                return False

        # 更新上一次检测结果
        self.last_detection[name] = conf
        return True

    def process_frame(self, image):
        """处理每一帧图像"""
        # 调整图像大小为显示尺寸
        image = cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        # 预处理图像
        pre_img = self.model.preprocess(image)

        t1 = time.time()
        pred, superimposed_img = self.model.predict(pre_img)
        t2 = time.time()
        use_time = t2 - t1
        print("推理时间: %.2f" % use_time)

        det = pred[0]

        if det is not None and len(det):
            det_info = self.model.postprocess(pred)
            for info in det_info:
                name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info['class_id']
                label = '%s %.1f%%' % (name, conf * 100)
                image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=self.colors[cls_id])

                # 检查是否需要保存检测结果
                if self.should_save_detection(name, conf):
                    try:
                        print(f"保存检测结果: {name}, 置信度: {conf:.3f}")
                        self.db_manager.save_detection(name, conf)
                    except Exception as e:
                        print(f"保存检测结果失败: {str(e)}")
                else:
                    print(f"跳过保存: {name}, 置信度: {conf:.3f}")

        # 将OpenCV图像转换为Qt图像并显示
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

        # 设置为拉伸填充整个 QLabel
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)  # 关键：允许拉伸填充

        # 缓存当前帧和检测结果
        self.current_frame = image.copy()
        self.last_detections = det_info

    def start_image_detection(self):
        """开始图片检测"""
        image_path = self.image_path_label.text()
        if image_path and image_path != "选择文件":
            image = cv_imread(image_path)
            if image is not None:
                self.process_frame(image)

    def stop_image_detection(self):
        # TODO: 实现停止图片检测功能
        pass

    def start_video_detection(self):
        """开始视频检测"""
        video_path = self.video_path_label.text()
        if video_path and video_path != "选择文件":
            if self.video_handler:
                self.stop_video_detection()

            self.video_handler = MediaHandler(fps=30)
            self.video_handler.frameReady.connect(self.process_frame)
            self.video_handler.setDevice(video_path)
            self.video_handler.startMedia()

    def stop_video_detection(self):
        """停止视频检测"""
        if self.video_handler:
            self.video_handler.stopMedia()
            self.video_handler = None
            self.result_label.setText("检测已停止")
            # 清空检测记录
            self.db_manager.last_save_time.clear()
            self.last_detection.clear()
            self.consecutive_count.clear()

    def start_camera(self):
        """开启摄像头"""
        if self.video_handler:
            self.stop_camera()

        self.video_handler = MediaHandler(fps=30)
        self.video_handler.frameReady.connect(self.process_frame)
        self.video_handler.setDevice(device=0)
        self.video_handler.startMedia()

    def stop_camera(self):
        """关闭摄像头"""
        self.stop_video_detection()

    def start_aianalyzer(self):
        """显示AI分析结果"""
        try:
            # 检查API密钥
            if not self.ai_analyzer.api_key:
                api_key, ok = QInputDialog.getText(
                    self,
                    "设置API密钥",
                    "请输入KiMi分析 API密钥：",
                    QLineEdit.Normal,
                    ""
                )
                if ok and api_key:
                    self.ai_analyzer.api_key = api_key
                    self.config['ai_api_key'] = api_key
                    self.save_config()
                else:
                    return

            # 获取最近的检测数据
            detection_data = self.get_recent_detections()

            # 显示加载提示
            self.result_label.setText("正在进行AI分析，请稍候...")
            self.ai_analysis_btn.setEnabled(False)

            # 开始异步分析
            analysis_thread = self.ai_analyzer.analyze_detection_data(detection_data)
            analysis_thread.analysis_complete.connect(self.handle_analysis_complete)
            analysis_thread.analysis_error.connect(self.handle_analysis_error)
            analysis_thread.finished.connect(self.handle_analysis_finished)

        except Exception as e:
            QMessageBox.warning(self, "错误", f"AI分析出错: {str(e)}")
            self.result_label.setText("AI分析失败")
            self.ai_analysis_btn.setEnabled(True)

    def update_frame(self):
        """更新画面"""
        if not hasattr(self, 'cap') or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # 处理帧
            self.process_frame(frame)
        else:
            # 如果读取失败，尝试重新连接
            self.result_label.setText("视频流读取失败，正在尝试重新连接...")
            self.reconnect_ipcamera()

    def send_aianalyzer(self):
        if not self.latest_ai_report:
            QMessageBox.warning(self, "提示", "请先完成一次A分析")
            return

        # 获取邮箱配置
        smtp_config = self.config.get("smtp_config", {})
        if not smtp_config:
            QMessageBox.warning(self, "错误", "未配置SMTP邮箱，请添加")
            return

        to_email, ok = QInputDialog.getText(self, "发送报告", "请输入接收邮箱")
        if not ok and to_email:
            return

        subject = f"输电线路部件AI分析报告 - {self.latest_ai_report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        context = self.latest_ai_report['ai_analysis']

        success = send_analysis_report(to_email, subject, context, smtp_config)
        if success:
            QMessageBox.information(self, "成功", "报告已发送到指定邮箱")
        else:
            QMessageBox.warning(self, "失败", "邮件发送失败，请检查配置")

    def show_records(self):
        """显示检测记录对话框"""
        dialog = DetectionRecordDialog(self.db_manager, self)
        dialog.exec()

    def get_recent_detections(self):
        """获取最近的检测数据"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()

        # 获取最近24小时的检测数据
        cursor.execute('''
                       SELECT timestamp, detection_type, confidence
                       FROM detection_results
                       WHERE timestamp >= datetime('now', '-24 hours')
                       ORDER BY timestamp
                       ''')

        records = cursor.fetchall()
        conn.close()

        # 格式化数据
        detection_data = "最近24小时检测记录：\n"
        for record in records:
            timestamp, detection_type, confidence = record
            detection_data += f"时间: {timestamp}, 类型: {detection_type}, 置信度: {confidence:.2%}\n"

        return detection_data

    def show_ai_analysis(self):
        """显示AI分析结果"""
        try:
            # 检查API密钥
            if not self.ai_analyzer.api_key:
                api_key, ok = QInputDialog.getText(
                    self,
                    "设置API密钥",
                    "请输入KiMi分析 API密钥：",
                    QLineEdit.Normal,
                    ""
                )
                if ok and api_key:
                    self.ai_analyzer.api_key = api_key
                    self.config['ai_api_key'] = api_key
                    self.save_config()
                else:
                    return

            # 获取最近的检测数据
            detection_data = self.get_recent_detections()

            # 显示加载提示
            self.result_label.setText("正在进行AI分析，请稍候...")
            self.ai_analysis_btn.setEnabled(False)

            # 开始异步分析
            analysis_thread = self.ai_analyzer.analyze_detection_data(detection_data)
            analysis_thread.analysis_complete.connect(self.handle_analysis_complete)
            analysis_thread.analysis_error.connect(self.handle_analysis_error)
            analysis_thread.finished.connect(self.handle_analysis_finished)

        except Exception as e:
            QMessageBox.warning(self, "错误", f"AI分析出错: {str(e)}")
            self.result_label.setText("AI分析失败")
            self.ai_analysis_btn.setEnabled(True)

    def handle_analysis_complete(self, result):
        """处理分析完成"""
        self.latest_ai_report = result
        dialog = AnalysisDialog(result, self)
        dialog.exec()

    def handle_analysis_error(self, error_msg):
        """处理分析错误"""
        QMessageBox.warning(self, "错误", f"AI分析失败: {error_msg}")
        self.result_label.setText("AI分析失败")

    def handle_analysis_finished(self):
        """分析完成后的清理工作"""
        self.ai_analysis_btn.setEnabled(True)
        self.result_label.setText("AI分析完成")

    # 一键存档
    def save_defect(self):
        """把当前帧原图 + JSON 存档"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "提示", "没有可保存的检测结果")
            return

        # 弹窗输入塔号
        tower, ok = QInputDialog.getText(self, "塔号", "请输入塔号：")
        if not ok or not tower.strip():
            return
        tower = tower.strip()

        # 生成时间戳
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = ARCHIVE_ROOT / f"{tower}_{ts}"
        save_dir.mkdir(exist_ok=True)

        # 保存原图
        origin_path = str(save_dir / "origin.png")
        cv2.imwrite(origin_path, self.current_frame)

        # 构造 JSON
        info = {
            "tower": tower,
            "timestamp": datetime.now().isoformat(),
            "detections": self.last_detections,  # 需要在 process_frame 里记录
            "ai_description": self.latest_ai_report.get("ai_analysis", "") if self.latest_ai_report else ""
        }
        json_path = str(save_dir / "meta.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        # 生成缩略图（200宽）
        thumb_path = str(save_dir / "thumb.jpg")
        im = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
        im.thumbnail((200, 200))
        im.save(thumb_path, quality=80)

        QMessageBox.information(self, "完成", f"已存档至\n{save_dir}")
        QRShareDialog(save_dir, self).exec()

    # 打开本地图库
    def open_gallery(self):
        dlg = GalleryDialog(self)
        dlg.item_double_clicked.connect(self.load_archive_back)
        dlg.exec()

    # 双击图库回显
    def load_archive_back(self, folder: Path):
        """把存档目录里的 origin.png 重新加载显示"""
        img_path = folder / "origin.png"
        if not img_path.exists():
            return
        img = cv2.imread(str(img_path))
        # self.display_frame(img)  # 重用现有显示函数
        # 如需要，可再读 meta.json 把检测框画回去（略）

    def start_voice_assistant(self):
        """启动语音助手"""
        try:
            # 确保数据库路径已正确设置
            if not os.path.exists(self.db_config['db_path']):
                QMessageBox.warning(self, "错误", "数据库文件不存在，请先进行一些检测")
                return

            self.voice_assistant.start()
            self.start_voice_btn.setEnabled(False)
            self.stop_voice_btn.setEnabled(True)
            self.result_label.setText("千巡已启动，每1分钟自动分析高压架输电线路部件状态")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动语音助手失败: {str(e)}")

    def stop_voice_assistant(self):
        """停止语音助手"""
        try:
            self.voice_assistant.stop()
            self.start_voice_btn.setEnabled(True)
            self.stop_voice_btn.setEnabled(False)
            self.result_label.setText("千巡已停止监测")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"停止语音助手失败: {str(e)}")


def main():
    app = QApplication(sys.argv)

    # 显示登陆界面
    from login_gui import LoginWindow
    login = LoginWindow()
    if login.exec() == QDialog.Accepted:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())


class GalleryDialog(QDialog):
    """缩略图瀑布流"""
    item_double_clicked = Signal(Path)  # 双击信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("本地缺陷图库")
        self.resize(600, 400)
        self.setStyleSheet("""
            QDialog { background:#121212; }
            QLabel { border:1px solid #0288d1; margin:4px; }
        """)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border:none; }")
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.scroll.setWidget(self.content)

        lay = QVBoxLayout(self)
        lay.addWidget(self.scroll)
        self.populate()

    def populate(self):
        """枚举 archive/* 并加载缩略图"""
        row = col = 0
        for item in sorted(ARCHIVE_ROOT.glob("*_*")):
            if not (item / "thumb.jpg").exists():
                continue
            lbl = QLabel()
            lbl.setFixedSize(180, 180)
            lbl.setScaledContents(True)
            lbl.setPixmap(QPixmap(str(item / "thumb.jpg")))
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.mouseDoubleClickEvent = lambda _, f=item: self.item_double_clicked.emit(f)
            self.grid.addWidget(lbl, row, col)
            col += 1
            if col == 3:
                col = 0
                row += 1


if __name__ == "__main__":
    main() 