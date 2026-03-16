#!/usr/bin/env python3
"""
visualizer package
多模态BCI平台可视化模块 - Tkinter版本
"""

from .signal_view import SignalView
from .stats_view import StatisticalAnalysisView
from .bar_view import BarView
from .plot_dialog import PlotDialog, quick_plot

__all__ = [
    'SignalView',
    'StatsView',
    'BarView',
    'PlotDialog',
    'quick_plot'
]