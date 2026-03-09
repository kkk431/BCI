import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QButtonGroup, QStackedWidget, QVBoxLayout)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap
from pathlib import Path

# 将'core'的上一级目录作为项目根目录添加到 sys.path 中，以便正确导入模块
start_path = Path(__file__).resolve().parent
for parent in [start_path] + list(start_path.parents):
    if parent.name == 'core':
        project_root = parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        break
else:
    raise RuntimeError("未找到名为 'core' 的目录")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置固定窗口大小
        self.setFixedSize(1440, 1024)
        self.setWindowTitle("主界面")

        # ========== 左侧导航栏容器 ==========
        self.left_nav = QWidget(self)
        self.left_nav.setGeometry(0, 0, 289, 1024)
        self.left_nav.setObjectName("left_nav")

        # ---------- 导航栏背景 ----------
        self.bg_label = QLabel(self.left_nav)
        self.bg_label.setGeometry(9, 9, 270, 1006)
        bg_path = os.path.join("core", "UI", "UI_resource", "Navigation", "Background.png")
        if os.path.exists(bg_path):
            pixmap = QPixmap(bg_path)
            # 缩放背景以适应标签大小（假设图片尺寸不完全匹配）
            self.bg_label.setPixmap(pixmap.scaled(270, 1006, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            self.bg_label.setScaledContents(True)
        else:
            self.bg_label.setText("背景图片缺失")
            self.bg_label.setStyleSheet("background-color: gray;")

        # ---------- 五个导航按钮 ----------
        # 按钮基本信息：名称、未选中图片文件名、选中图片文件名、Y坐标
        buttons_info = [
            ("首页", "Home_Button.png", "Home_Button.png", 231),
            ("预处理", "Preprocessing_Button.png", "Preprocessing_Button.png", 323),
            ("特征提取", "Feature_Extraction_Button.png", "Feature_Extraction_Button.png", 415),
            ("统计分析", "Statistical_Analysis_Button.png", "Statistical_Analysis_Button.png", 507),
            ("可视化", "Virtualization_Button.png", "Virtualization_Button.png", 599)
        ]

        self.buttons = []
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)   # 同一时刻只有一个按钮被选中

        # 获取按钮图片尺寸（假设所有按钮图片尺寸一致）
        example_path = os.path.join("core", "UI", "UI_resource", "Navigation", "Buttons", "Unselected", "Home_Button.png")
        if os.path.exists(example_path):
            pix = QPixmap(example_path)
            btn_width = pix.width()
            btn_height = pix.height()
        else:
            # 如果图片不存在，使用默认尺寸（后续可手动调整）
            btn_width, btn_height = 200, 80
            print("警告：未找到示例图片，使用默认按钮尺寸 200x80")

        for idx, (name, unsel_file, sel_file, y) in enumerate(buttons_info):
            btn = QPushButton(self.left_nav)
            btn.setGeometry(39, y, btn_width, btn_height)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)

            # 构建未选中和选中图片的完整路径
            unsel_path = os.path.join("core", "UI", "UI_resource", "Navigation", "Buttons", "Unselected", unsel_file)
            sel_path = os.path.join("core", "UI", "UI_resource", "Navigation", "Buttons", "Selected", sel_file)

            # 样式表中的路径需要使用正斜杠，且如果是本地文件无需额外协议
            unsel_path = unsel_path.replace('\\', '/')
            sel_path = sel_path.replace('\\', '/')

            # 设置样式表：未选中状态使用未选中图片，选中状态使用选中图片
            style = f"""
            QPushButton {{
                background-image: url({unsel_path});
                background-repeat: no-repeat;
                background-position: center;
                border: none;
            }}
            QPushButton:checked {{
                background-image: url({sel_path});
            }}
            """
            btn.setStyleSheet(style)

            self.buttons.append(btn)
            self.button_group.addButton(btn, idx)   # 将索引作为按钮 ID

        # 默认选中“首页”按钮
        self.buttons[0].setChecked(True)

        # ========== 右侧功能区（堆叠窗口） ==========
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.setGeometry(289, 0, 1151, 1024)

        # 创建五个场景页面（预留接口，目前仅放置一个标签）
        scene_names = ["首页", "预处理", "特征提取", "统计分析", "可视化"]
        self.pages = []
        for name in scene_names:
            page = QWidget()
            layout = QVBoxLayout(page)
            label = QLabel(f"这是 {name} 场景")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self.stacked_widget.addWidget(page)
            self.pages.append(page)

        # 默认显示首页场景
        self.stacked_widget.setCurrentIndex(0)

        # 连接按钮组的点击信号（传递按钮 ID）
        self.button_group.buttonClicked[int].connect(self.on_nav_button_clicked)

    def on_nav_button_clicked(self, btn_id):
        """导航按钮点击时，切换右侧场景"""
        self.stacked_widget.setCurrentIndex(btn_id)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())