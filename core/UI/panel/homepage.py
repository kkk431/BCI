import os
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPixmap


class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(1151, 1024)  # 固定大小，与主界面右侧区域一致
        self.initUI()

    def initUI(self):
        # 定义每个组件的信息: (名称, 相对路径, x, y)
        components = [
            ("Files",      "Files.png",      610, 267),
            ("Group",      "Group.png",      610, 813),
            ("ProjectName","Project_Name.png",7,   142),
            ("Search",     "Search.png",     610, 182),
            ("User",       "User.png",       7,   665),
            ("Welcome",    "Welcome.png",    7,   7)
        ]

        # 基础路径
        base_dir = os.path.join("core", "UI", "UI_resource", "Homepage")

        for name, filename, x, y in components:
            # 构造完整路径
            full_path = os.path.join(base_dir, filename)

            label = QLabel(self)
            if os.path.exists(full_path):
                pix = QPixmap(full_path)
                # 使用图片原始尺寸
                label.setPixmap(pix)
                label.setGeometry(x, y, pix.width(), pix.height())
            else:
                # 图片缺失时显示占位信息
                label.setText(f"{name}\n(图片缺失)")
                label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
                # 设置一个默认大小（例如 200x100），可根据实际调整
                default_w, default_h = 200, 100
                label.setGeometry(x, y, default_w, default_h)
                label.setAlignment(QLabel.AlignCenter)

            label.setObjectName(f"home_{name}")


# 用于单独测试该页面
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = HomePage()
    window.setWindowTitle("首页")
    window.show()
    sys.exit(app.exec_())