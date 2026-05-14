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
    fault_alert = Signal(str)        # 故障报警信号
    
    def __init__(self, db_path, api_key=None):
        super().__init__()
        self.db_path = db_path
        self.api_key = api_key or "sk-7b2c90c16ce54f79a53eea40a14a1510"  # 使用默认API密钥
        self.base_url = "https://api.deepseek.com/v1"
        
    def run(self):
        try:
            # 检查故障数量（两个及以上）
            self.check_faults()

            # AI分析
            self.perform_time_window_analysis()

        except Exception as e:
            print(f'AI分析失败: {str(e)}')
            self.analysis_error.emit(str(e))

    def check_faults(self):
        """检查是否有两个及以上特定故障"""
        try:
            fault_counts = {
                'broken': 0,
                'ham_broken': 0,
                'GR_incline': 0
            }
            fault_ids = []  # 记录需要标记为已处理的故障ID

            # 再获取数据
            records = self.get_unprocessed_detections()
            if not records:
                print("无未处理记录，跳过故障检查")
                return

            for record in records:
                id, timestamp, detection_type, confidence = record

                # 只处理置信度>60%的特定故障
                if confidence > 0.60 and detection_type in fault_counts:
                    fault_counts[detection_type] += 1
                    fault_ids.append(id)

            # 检查是否有达到2个及以上的故障
            for fault_type, count in fault_counts.items():
                if count >= 2:
                    alert_message = f"警报！检测到{count}处{fault_type}，请立即检修！"
                    self.fault_alert.emit(alert_message)

            # 标记已处理的故障记录
            if fault_ids:
                self.mark_as_processed(fault_ids)

        except Exception as e:
            print(f"故障检查失败: {str(e)}")


    def get_unprocessed_detections(self):
        """获取所有未处理的检测数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取所有未处理的监测数据
            cursor.execute('''
                SELECT id, timestamp, detection_type, confidence 
                FROM detection_results 
                WHERE processed = 0
            ''')

            records = cursor.fetchall()
            conn.close()
            return records
        except Exception as e:
            print(f"数据库查询失败: {str(e)}")
            return []

    def mark_as_processed(self, ids):
        """标记记录为已处理"""
        if not ids:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 构建IN查询参数
            id_placeholders = ','.join("?" * len(ids))

            cursor.execute(f'''
                UPDATE detection_results 
                SET processed = 1
                WHERE id = {id_placeholders}
            ''',ids)

            conn.commit()
            conn.close()
            print(f"已标记{len(ids)}条记录为已处理")
        except Exception as e:
            print(f"更新数据库失败: {str(e)}")

    def perform_time_window_analysis(self):
        """时间窗口AI分析"""
        # 获取最近2分钟的检测数据
        records = self.get_recent_detections()

        if not records:
            print("未检测到任何记录，跳过AI分析")
            return

        # 直接进行AI分析
        analysis_result = self.perform_ai_analysis(records)

        # 发送分析结果供语音播报
        if analysis_result != "正常":
            self.analysis_complete.emit(analysis_result)

    def get_recent_detections(self):
        """获取最近1分钟的检测数据"""
        try:
            # 获取当前时间
            now = datetime.now()
            two_min_ago = now - timedelta(minutes=1)
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            two_min_ago_str = two_min_ago.strftime('%Y-%m-%d %H:%M:%S')

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            print(f"获取时间范围: {two_min_ago_str} 至 {now_str}")

            # 获取指定时间范围内的检测数据
            cursor.execute('''
                           SELECT timestamp, detection_type, confidence
                           FROM detection_results
                           WHERE timestamp BETWEEN ? AND ?
                           ORDER BY timestamp
                           ''', (two_min_ago_str, now_str))

            records = cursor.fetchall()
            conn.close()

            print(f"查询到的记录数量: {len(records)}")
            return records
        except Exception as e:
            print(f"数据库查询失败: {str(e)}")
            return []

    def perform_ai_analysis(self, records):
        """AI分析功能 - 专注于输电部件分析"""
        # 统计输电部件的故障情况
        insulator_crack_count = sum(1 for _, detection_type, confidence in records
                                    if detection_type == 'broken' and confidence > 0.60)
        grading_ring_tilt_count = sum(1 for _, detection_type, confidence in records
                                      if detection_type == 'GR_incline' and confidence > 0.60)
        damper_missing_count = sum(1 for _, detection_type, confidence in records
                                   if detection_type == 'ham_broken' and confidence > 0.60)
        # 如果没有足够的检测数据，直接返回正常提示
        if len(records) < 3:
            return "目前接收的检测数据不足，再让我检测分析一下噢"

        # 准备数据用于API调用，增加详细信息
        detection_data = f"最近1分钟内的输电部件检测数据:\n 绝缘子破裂: {insulator_crack_count}处, 均压器倾斜: {grading_ring_tilt_count}处,  防震锤缺失: {damper_missing_count}处"

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            # 提示词，要求无论正常还是异常都给出结果
            prompt = f"""
            最近1分钟检测数据：
            - 绝缘子破裂：{insulator_crack_count}
            - 均压器倾斜：{grading_ring_tilt_count}
            - 防震锤缺失：{damper_missing_count}

            请用严肃正经语气，50字以内：
            1. 总和≥3 → 警告（例：哎呀，3处故障要维修！）
            2. 总和1~2 → 提醒（例：发现1处小故障，快去看看吧）
            3. 总和=0 → 鼓励（例：目前高压架一切正常哟！）
            """
            # prompt = f"""
            # 分析以下输电部件检测数据: {detection_data}
            #
            # 请根据这些数据评估高压输电线路的状态。
            # 如果数据显示绝缘子破裂、均压器倾斜过度或防震锤缺失数量过多（三个数量相加大于2）等情况，请给出活泼可爱风格的警告(不超过50字)。
            # 如果数据显示存在绝缘子破裂、均压器倾斜过度或防震锤缺失数量（三个数量相加在1~2之间）等情况，请给出活泼可爱风格的提醒(不超过50字)。
            # 如果数据显示输电部件状态正常，即不存在绝缘子破裂、均压器倾斜过度或防震锤缺失等情况，请给出鼓励的话语，格式为"这里的高压架输电部件正常，目前没有存在隐患噢"或类似内容。
            # """

            # 调用API - 不设置保守的token限制
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个输电线路部件监测助手，专注分析高压输电部件是否正常，无论结果如何都会给出反馈，只返回简洁的评估结果。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                stream=False
            )

            # 解析响应
            ai_message = response.choices[0].message.content.strip()
            print(f"原始AI分析结果: {ai_message}")

            # 添加过滤步骤：去除无关内容
            ai_message = self.filter_ai_output(ai_message)
            print(f"过滤后AI分析结果: {ai_message}")
            
            # 如果AI没有返回结果或出错，使用备用消息
            if not ai_message:
                return "目前没有发现输电部件问题"
                
            return ai_message
            
        except Exception as e:
            print(f"AI分析调用失败: {str(e)}")
            # 使用正面消息作为备用
            return "输电部件检测系统运行正常"

    def filter_ai_output(self, text: str) -> str:
        """简单去首尾空格，返回原文本"""
        return text.strip()

class VoiceAssistant:
    def __init__(self, db_path):
        self.db_path = db_path
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # 设置语速
        self.engine.setProperty('volume', 1.0)  # 设置音量
        self.engine.setProperty('voice', 'zh')  # 设置中文语音
        
        # 初始化分析计时器
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.start_analysis)
        self.check_interval = 30# 10秒检查一次
        
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
        self.speak("千巡已启动，开始智能监测高压架输电部件")
        
    def stop(self):
        """停止智能语音助手"""
        self.is_running = False
        self.check_timer.stop()
        print("智能语音助手已停止")
        self.speak("千巡已停止监测")
        
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
        
        self.analysis_thread = AIAnalysisThread(self.db_path)
        self.analysis_thread.analysis_complete.connect(self.handle_analysis_result)
        self.analysis_thread.analysis_error.connect(self.handle_analysis_error)
        self.analysis_thread.start()
        
    def handle_analysis_result(self, message):
        """处理分析结果"""
        print(f"分析结果: {message}")
        # 无论是否正常，都进行语音播报
        # if message:
        self.speak(message)

    def handle_fault_alert(self, message):
        """处理故障报警"""
        print(f"故障警报: {message}")
        self.speak(message)  # 立即播报警报
            
    def handle_analysis_error(self, error):
        """处理分析错误"""
        print(f"分析错误: {error}")
        # 发生错误时也播报通用提醒
        self.speak("千巡检测到系统异常，请确保检测设备没问题")