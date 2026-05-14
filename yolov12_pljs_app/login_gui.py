import os
import json
import hashlib
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# ---------- 工具函数 ----------
def hash_pwd(password: str) -> str:
    """简单哈希，用于本地存储"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    """加载用户文件"""
    if not os.path.exists("users.json"):
        return {}
    with open("users.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users: dict):
    """保存用户文件"""
    with open("users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


# ---------- 注册对话框 ----------
class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("用户注册")
        self.setFixedSize(300, 230)
        self.setStyleSheet(self.style_sheet())
        self.init_ui()

    def style_sheet(self):
        return """
            QDialog { background-color:#121212; color:#e0e0e0; }
            QLabel { font-size:13px; }
            QLineEdit {
                background:#2c2c2c; border:1px solid #424242; border-radius:4px;
                padding:6px; color:#e0e0e0;
            }
            QPushButton {
                background:#0288d1; color:#fff; border:none; border-radius:4px;
                padding:8px 16px; font-weight:bold;
            }
            QPushButton:hover { background:#039be5; }
            QPushButton:pressed { background:#026baa; }
        """

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("用户名（3~16位）")
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setPlaceholderText("密码（6~20位）")
        self.confirm_edit = QLineEdit()
        self.confirm_edit.setEchoMode(QLineEdit.Password)
        self.confirm_edit.setPlaceholderText("确认密码")

        reg_btn = QPushButton("注册")
        reg_btn.clicked.connect(self.do_register)

        layout.addWidget(QLabel("用户名："))
        layout.addWidget(self.user_edit)
        layout.addWidget(QLabel("密码："))
        layout.addWidget(self.pwd_edit)
        layout.addWidget(QLabel("确认密码："))
        layout.addWidget(self.confirm_edit)
        layout.addWidget(reg_btn)

    def do_register(self):
        username = self.user_edit.text().strip()
        pwd = self.pwd_edit.text()
        confirm = self.confirm_edit.text()

        if len(username) < 3 or len(username) > 16:
            QMessageBox.warning(self, "提示", "用户名长度应为3~16位")
            return
        if len(pwd) < 6 or len(pwd) > 20:
            QMessageBox.warning(self, "提示", "密码长度应为6~20位")
            return
        if pwd != confirm:
            QMessageBox.warning(self, "提示", "两次密码不一致")
            return

        users = load_users()
        if username in users:
            QMessageBox.warning(self, "提示", "用户名已存在")
            return

        users[username] = {
            "password": hash_pwd(pwd),
            "register_time": datetime.now().isoformat(timespec="seconds")
        }
        save_users(users)
        QMessageBox.information(self, "成功", "注册成功，请返回登录")
        self.accept()


# ---------- 登录对话框 ----------
class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("天轨智寻 - 登录")
        self.setFixedSize(400, 320)
        self.setStyleSheet(self.style_sheet())
        self.init_ui()
        self.load_saved()

    def style_sheet(self):
        return """
            QDialog { background-color:#121212; color:#e0e0e0; }
            QLabel { color:#e0e0e0; }
            QLineEdit { background:#2c2c2c; border:1px solid #424242; border-radius:4px;
                        padding:8px; color:#e0e0e0; font-size:14px; }
            QPushButton { background:#0288d1; color:#fff; border:none; border-radius:4px;
                          padding:10px 20px; font-weight:bold; }
            QPushButton:hover { background:#039be5; }
            QCheckBox { color:#bdbdbd; font-size:12px; }
        """

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        title = QLabel("天轨智寻")
        title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)
        title.setStyleSheet("color:#00e5ff;")

        subtitle = QLabel("输电线路部件检测系统")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:#bdbdbd; font-size:12px;")

        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("用户名")
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setPlaceholderText("密码")

        self.remember_check = QCheckBox("记住密码")

        login_btn = QPushButton("登录")
        reg_btn = QPushButton("注册新用户")

        login_btn.clicked.connect(self.do_login)
        reg_btn.clicked.connect(self.do_register)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)
        layout.addWidget(self.user_edit)
        layout.addWidget(self.pwd_edit)
        layout.addWidget(self.remember_check)
        layout.addWidget(login_btn)
        layout.addWidget(reg_btn)

    def load_saved(self):
        if os.path.exists("credentials.json"):
            try:
                with open("credentials.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.user_edit.setText(data.get("username", ""))
                    self.pwd_edit.setText(data.get("password", ""))
                    self.remember_check.setChecked(True)
            except:
                pass

    def save_credential(self, username, password):
        if self.remember_check.isChecked():
            with open("credentials.json", "w", encoding="utf-8") as f:
                json.dump({"username": username, "password": password}, f)
        else:
            if os.path.exists("credentials.json"):
                os.remove("credentials.json")

    def do_login(self):
        username = self.user_edit.text().strip()
        pwd = self.pwd_edit.text()

        users = load_users()
        if username not in users:
            QMessageBox.warning(self, "错误", "用户不存在")
            return
        if users[username]["password"] != hash_pwd(pwd):
            QMessageBox.warning(self, "错误", "密码错误")
            return

        self.save_credential(username, pwd)
        self.accept()

    def do_register(self):
        reg = RegisterDialog(self)
        reg.exec()


# ---------- 主入口 ----------
if __name__ == "__main__":
    app = QApplication([])
    login = LoginWindow()
    if login.exec() == LoginWindow.Accepted:
        from main_window import MainWindow  # 确保在同一目录
        main_win = MainWindow()
        main_win.show()
        app.exec()
