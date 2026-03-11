import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("主界面")
        self.setFixedSize(1440, 1024)          # 固定窗口大小

        # 初始化按钮列表
        self.buttons = []
        # 加载所有静态图片
        self.load_images()

        # 创建五个导航按钮
        self.create_buttons()

    def load_images(self):
        """放置所有静态图片（背景及首页元素）"""
        # 图片路径与坐标 (已手动计算好坐标)
        images = [
            ("core/UI/UI_resource/Navigation/Background.png", (9, 9)),
            ("core/UI/UI_resource/Homepage/Files.png", (899, 267)),      # 610+289=899
            ("core/UI/UI_resource/Homepage/Group.png", (899, 813)),      # 610+289=899
            ("core/UI/UI_resource/Homepage/Project_Name.png", (296, 142)), # 7+289=296
            ("core/UI/UI_resource/Homepage/Search.png", (899, 182)),     # 610+289=899
            ("core/UI/UI_resource/Homepage/User.png", (296, 665)),       # 7+289=296
            ("core/UI/UI_resource/Homepage/Welcome.png", (296, 7))       # 7+289=296
        ]

        for path, pos in images:
            if os.path.exists(path):
                pixmap = QPixmap(path)
                label = QLabel(self)
                label.setPixmap(pixmap)
                label.setFixedSize(pixmap.size())   # 保证标签大小与图片一致
                label.move(pos[0], pos[1])
            else:
                print(f"警告：图片不存在 - {path}")

    def create_buttons(self):
        """创建五个导航按钮，并连接点击信号"""
        buttons_info = [
            ("首页", (39, 231), "core/UI/UI_resource/Navigation/Buttons/Selected/Home_Button.png"),
            ("预处理", (39, 323), "core/UI/UI_resource/Navigation/Buttons/Unselected/Preprocessing_Button.png"),
            ("特征提取", (39, 415), "core/UI/UI_resource/Navigation/Buttons/Unselected/Feature_Extraction_Button.png"),
            ("统计分析", (39, 507), "core/UI/UI_resource/Navigation/Buttons/Unselected/Statistical_Analysis_Button.png"),
            ("可视化", (39, 599), "core/UI/UI_resource/Navigation/Buttons/Unselected/Virtualization_Button.png")
        ]

        for text, pos, img_path in buttons_info:
            if os.path.exists(img_path):
                btn = QPushButton(self)
                btn.setToolTip(text)

                # 正确设置样式表：移除边框，设置背景图片
                btn.setStyleSheet(f"""
                    QPushButton {{
                        border: none;
                        border-image: url({img_path});
                    }}
                """)

                # 设置按钮大小与图片一致
                pixmap = QPixmap(img_path)
                btn.setFixedSize(pixmap.size())
                btn.move(pos[0], pos[1])

                # 确保按钮在最上层（防止被其他控件遮挡）
                btn.raise_()

                btn.clicked.connect(lambda checked, t=text: self.on_button_clicked(t))
                self.buttons.append(btn)
            else:
                print(f"警告：按钮图片不存在 - {img_path}")

    def on_button_clicked(self, button_name):
        """按钮点击的槽函数 —— 预留场景切换接口"""
        print(f"【场景切换】点击了 {button_name} 按钮")
        # TODO: 根据 button_name 切换到对应的场景
        # 例如：
        # if button_name == "首页":
        #     self.switch_to_home()
        # elif button_name == "预处理":
        #     self.switch_to_preprocessing()
        # ...

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())