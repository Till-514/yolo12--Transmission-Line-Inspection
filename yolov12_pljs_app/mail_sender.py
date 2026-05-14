import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

def send_analysis_report(to_email: str, subject: str, content: str, smtp_config: dict):
    """
    发送AI分析报告到指定邮箱
    :param to_email: 接收邮箱
    :param subject: 邮件主题
    :param content: 报告内容
    :param smtp_config: 包含 'server', 'port', 'user', 'password' 的字典
    """

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['user']
        msg['To'] = to_email
        msg['Subject'] = subject

        # 添加正文
        msg.attach(MIMEText(content, 'plain', 'utf-8'))

        # 连接并发送
        with smtplib.SMTP_SSL(smtp_config['server'], smtp_config['port']) as server:
            server.login(smtp_config['user'], smtp_config['password'])
            server.sendmail(smtp_config['user'], to_email, msg.as_string())
            server.quit()
        print("✅ 邮件发送成功")
        return True

    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False