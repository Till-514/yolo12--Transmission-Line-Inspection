import os
import json
import socket
import qrcode
from pathlib import Path
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices, QImage

LAN_PORT = 8001     # 本地 HTTP 端口，可改
ARCHIVE_ROOT = Path("archive").resolve()

def get_lan_ip():
    """取本机局域网 IP"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def build_share_link(folder: Path) -> str:
    """
    folder: archive/塔号_20240807_143022
    返回形如 http://<ip>:8000/archive/塔号_20240807_143022/origin.png
    """
    ip = get_lan_ip()
    url_path = str(folder / "origin.png").replace("\\", "/")
    return f"http://{ip}:{LAN_PORT}/{url_path.lstrip('/')}"

def generate_qr(text: str) -> QPixmap:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=6, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#000", back_color="#fff")
    # 转 QPixmap
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return QPixmap().fromImage(QImage.fromData(buf.getvalue()))

class QRShareDialog(QDialog):
    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("缺陷二维码分享")
        self.setFixedSize(300, 400)

        link = build_share_link(folder)
        qr_pix = generate_qr(link)
        self.folder = folder

        self.label = QLabel()
        self.label.setPixmap(qr_pix)
        self.label.setAlignment(Qt.AlignCenter)

        self.link_edit = QLineEdit(link)
        self.link_edit.setReadOnly(True)

        copy_btn = QPushButton("复制链接")
        copy_btn.clicked.connect(lambda: self.link_edit.selectAll() or self.link_edit.copy())

        open_btn = QPushButton("浏览器打开")
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(link)))

        lay = QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(self.link_edit)
        lay.addWidget(copy_btn)
        lay.addWidget(open_btn)
        save_qr_btn = QPushButton("保存二维码")
        save_qr_btn.clicked.connect(self.save_qr)
        lay.addWidget(save_qr_btn)


    def save_qr(self):
        """把当前二维码保存为 PNG"""
        # 默认文件名：存档文件夹名_qr.png
        default_name = f"{self.folder.name}_qr.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存二维码",
            str(Path.home() / default_name),
            "PNG (*.png)"
        )
        if file_path:
            self.label.pixmap().save(file_path, "PNG")
            QMessageBox.information(self, "完成", f"二维码已保存至\n{file_path}")
