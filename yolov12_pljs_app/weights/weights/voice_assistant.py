import pyttsx3
import sqlite3
from datetime import datetime, timedelta
from PySide6.QtCore import QTimer, QThread, Signal
from openai import OpenAI

class VoiceThread(QThread):
    finished = Signal()  # 播报完成信号
    
    def __init__(self, engine, text):
        super().__init__()
        self.engine = engine
        self.text = text
        
    def run(self):
        try:
            self.engine.say(self.text)
            self.engine.runAndWait()
            self.finished.emit()
        except Exception as e:
            print(f"语音播报失败: {str(e)}")

class AIAnalysisThread(QThread):
    analysis_complete = Signal(str)  # 分析完成信号，发送分析结果用于语音播报
    analysis_error = Signal(str)     # 错误信号
    
    def __init__(self, db_path, api_key=None):
        super().__init__()
        self.db_path = db_path
        self.api_key = api_key or ""  # 使用默认API密钥
        self.base_url = "https://api.kimiai.com"
        
    def run(self):
        try:
            # 获取最近2分钟的检测数据
            records = self.get_recent_detections()
            
            if not records or len(records) == 0:
                print("未检测到任何记录，跳过KiMi分析")
                return
                
            # 直接进行AI分析，不使用阈值判断
            analysis_result = self.perform_ai_analysis(records)
            
            # 发送分析结果供语音播报
            if analysis_result != "正常":
                self.analysis_complete.emit(analysis_result)
                
        except Exception as e:
            print(f"AI分析失败: {str(e)}")
            self.analysis_error.emit(str(e))
            
    def get_recent_detections(self):
        """获取最近1分钟的检测数据"""
        try:
            # 获取当前时间
            now = datetime.now()
            # 计算2分钟前的时间
            two_min_ago = now - timedelta(minutes=2)
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            two_min_ago_str = two_min_ago.strftime('%Y-%m-%d %H:%M:%S')
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print(f"获取时间范围: {two_min_ago_str} 至 {now_str}")
            
            # 获取指定时间范围内的检测数据（从当前时间向前推2分钟）
            cursor.execute('''
                SELECT timestamp, detection_type, confidence 
                FROM detection_results 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (two_min_ago_str, now_str))
            
            records = cursor.fetchall()
            conn.close()
            
            # 添加调试信息
            print(f"查询到的记录数量: {len(records)}")
            for record in records:
                print(f"记录: 时间={record[0]}, 类型={record[1]}, 置信度={record[2]:.3f}")
                
            return records
        except Exception as e:
            print(f"数据库查询失败: {str(e)}")
            return []
    
    def perform_ai_analysis(self, records):
        """使用AI进行分析"""
        # 统计打哈欠和闭眼次数用于AI分析
        yawn_count = sum(1 for _, detection_type, _ in records if detection_type == 'yawning')
        eyes_closed_count = sum(1 for _, detection_type, _ in records if detection_type == 'eyes_closed')
        
        # 如果没有足够的检测数据，直接返回正常提示
        if len(records) < 3:
            return "目前接收的检测数据不足，再让我检测分析一下噢"
        
        # 准备数据用于API调用，增加详细信息
        detection_data = f"最近1分钟内的驾驶检测数据:\n打哈欠次数: {yawn_count}次, 闭眼次数: {eyes_closed_count}次"
        
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            # 提示词，要求无论正常还是异常都给出结果
            prompt = f"""
            分析以下驾驶检测数据: {detection_data}
            
            如果1分钟内绝缘子出现破裂或者均压器倾斜度数过大（置信度超过65%）或者防震锤断裂，判定为危险状态，请给出活泼可爱风格的提醒(不超过50字)。
            如果数据显示驾驶状态正常，请给出鼓励的话语，格式为"这里的高压架输电部件正常，目前没有存在隐患噢"或类似内容。
            不要只回复"正常"，一定要给出完整的活泼可爱风格的回答。
            """

            # 调用API - 不设置保守的token限制
            response = client.chat.completions.create(
                model="kimiai-chat",
                messages=[
                    {"role": "system", "content": "你是一个可爱活泼的驾驶助手，专注分析输电部件正常与否，无论结果如何都会给出反馈。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                stream=False
            )

            # 解析响应
            ai_message = response.choices[0].message.content.strip()
            print(f"AI分析结果: {ai_message}")
            
            # 如果AI没有返回结果或出错，使用备用消息
            if not ai_message:
                return "目前没有发现任何问题"
                
            return ai_message
            
        except Exception as e:
            print(f"AI分析调用失败: {str(e)}")
            # 使用正面消息作为备用
            return "目前没有发现任何问题"

class VoiceAssistant:
    def __init__(self, db_path, api_key=None):
        self.db_path = db_path
        self.api_key = api_key or ""  # 使用默认API密钥
        self.base_url = "https://api.kimiai.com/v1"  # 替换为实际的API基础URL
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # 设置语速
        self.engine.setProperty('volume', 1.0)  # 设置音量
        self.engine.setProperty('voice', 'zh')  # 设置中文语音
        
        # 初始化分析计时器
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.start_analysis)
        self.check_interval = 30  # 2分钟检查一次
        
        # 语音线程和分析线程
        self.voice_thread = None
        self.analysis_thread = None
        
        # 是否正在运行
        self.is_running = False
        
        # 记录上次分析时间
        self.last_analysis_time = None
        
    def start(self):
        """启动智能语音助手"""
        self.is_running = True
        # 立即执行一次分析
        self.start_analysis()
        # 设置定时器，精确到2分钟间隔
        self.last_analysis_time = datetime.now()
        next_analysis = self.check_interval - (self.last_analysis_time.second % self.check_interval)
        print(f"智能语音助手已启动，下次分析将在{next_analysis}秒后进行")
        
        # 设置定时器为整点触发
        self.check_timer.start(self.check_interval * 1000)  # 转换为毫秒
        self.speak("KiMi分析已启动，开始智能监测高压架输电部件")
        
    def stop(self):
        """停止智能语音助手"""
        self.is_running = False
        self.check_timer.stop()
        print("智能语音助手已停止")
        self.speak("KiMi分析已停止监测")
        
    def speak(self, text):
        """语音播报（多线程）"""
        print(f"准备播报: {text}")
        if self.voice_thread and self.voice_thread.isRunning():
            print("上一条语音正在播报中...")
            return
            
        self.voice_thread = VoiceThread(self.engine, text)
        self.voice_thread.start()
        print("语音播报已开始")
        
    def start_analysis(self):
        """开始分析检测数据"""
        current_time = datetime.now()
        
        if not self.is_running:
            print("语音助手未运行，跳过分析")
            return
            
        if self.analysis_thread and self.analysis_thread.isRunning():
            print("上一次分析尚未完成，跳过本次分析")
            return
            
        # 记录本次分析时间
        self.last_analysis_time = current_time
        print(f"\n开始AI分析输电部件的问题... 当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.analysis_thread = AIAnalysisThread(self.db_path, self.api_key)
        self.analysis_thread.analysis_complete.connect(self.handle_analysis_result)
        self.analysis_thread.analysis_error.connect(self.handle_analysis_error)
        self.analysis_thread.start()
        
    def handle_analysis_result(self, message):
        """处理分析结果"""
        print(f"分析结果: {message}")
        # 无论是否正常，都进行语音播报
        if message:
            self.speak(message)
            
    def handle_analysis_error(self, error):
        """处理分析错误"""
        print(f"分析错误: {error}")
        # 发生错误时也播报通用提醒
        self.speak("KiMi分析检测到系统异常，请确保检测设备没问题")
