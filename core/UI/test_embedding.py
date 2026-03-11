import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,
                             QStackedWidget, QHBoxLayout, QVBoxLayout)
from PyQt5.QtCore import Qt

# 导入我们写的统计分析面板
from panel.analysis_panel import AnalysisPanel


class SimpleTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("简易嵌入测试器")
        self.resize(1400, 900)

        # 主布局：左右结构
        main_layout = QHBoxLayout(self)

        # 1. 左侧：简易导航栏
        left_nav = QWidget()
        left_nav.setFixedWidth(200)
        left_nav.setStyleSheet("background-color: #f0f0f0;")
        nav_layout = QVBoxLayout(left_nav)

        # 两个测试按钮
        self.btn_page1 = QPushButton("页面 1 (占位)")
        self.btn_page2 = QPushButton("👉 统计分析面板")
        self.btn_page2.setStyleSheet("font-weight: bold; background-color: #d0e0ff;")

        nav_layout.addWidget(self.btn_page1)
        nav_layout.addWidget(self.btn_page2)
        nav_layout.addStretch()

        # 2. 右侧：堆叠窗口
        self.stacked = QStackedWidget()

        # 页面1：占位
        page1 = QWidget()
        page1.setStyleSheet("background-color: white;")
        layout1 = QVBoxLayout(page1)
        layout1.addStretch()
        label1 = QPushButton("这是占位页面\n请点击左侧按钮切换")
        label1.setEnabled(False)
        label1.setStyleSheet("font-size: 20px; border: none;")
        layout1.addWidget(label1)
        layout1.addStretch()

        # 页面2：我们的统计分析面板
        page2 = AnalysisPanel()

        self.stacked.addWidget(page1)
        self.stacked.addWidget(page2)

        # 组装
        main_layout.addWidget(left_nav)
        main_layout.addWidget(self.stacked)

        # 连接信号
        self.btn_page1.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.btn_page2.clicked.connect(lambda: self.stacked.setCurrentIndex(1))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleTestWindow()
    window.show()
    sys.exit(app.exec_())