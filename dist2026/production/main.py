import sys
import os
from pathlib import Path


DIST_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(DIST_DIR))
CONFIG_PATH = DIST_DIR / "config" / "config.ini"
DEVICE_LOCATIONS_CONFIG_PATH = CONFIG_PATH
OUTPUT_DIR = DIST_DIR / "output"
MODEL_OUTPUT_DIR = DIST_DIR / "model_output"

# 设置PyQt5插件路径 - 支持conda和venv环境
def setup_qt_plugins():
    try:
        # 优先使用Qt自带的路径检测
        from PyQt5.QtCore import QLibraryInfo
        if hasattr(QLibraryInfo, 'location'):
            qt_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
            if os.path.exists(qt_path):
                return qt_path
    except:
        pass
    
    # 回退方案：检测conda环境
    import sys
    if 'conda' in sys.prefix.lower() or 'anaconda' in sys.prefix.lower():
        # conda环境路径
        conda_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
        if os.path.exists(conda_path):
            return conda_path
    
    # 最后回退到.venv路径
    venv_path = os.path.join(str(DIST_DIR), '.venv', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
    if os.path.exists(venv_path):
        return venv_path
    
    # 如果都找不到，返回默认路径
    return os.path.join(str(DIST_DIR), '.venv', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = setup_qt_plugins()

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QCheckBox, QProgressDialog,
    QLabel, QTableWidget, QTableWidgetItem, QHBoxLayout, QMessageBox, QDialog, QTabWidget, QFormLayout, 
    QRadioButton, QButtonGroup, QInputDialog, QLineEdit, QGroupBox
)
from PyQt5.QtGui import QFont, QColor, QKeySequence
from PyQt5.QtCore import Qt, QEvent
# Mock imports for demonstration purposes
from data_fetcher.tolocation import DataFetcher
from production.train import DeviceModelTrainer
from data_fetcher.config_util import Config
from production.humidity_stage_ensemble_predictor import HumidityStageEnsemblePredictor


def calculate_battery_percentage(voltage):
    # 电压和百分比对应表
    voltage_to_percentage = [
        (4.2, 100),(4.06, 90),(3.98, 80),(3.92, 70),(3.87, 60),(3.82, 50),
        (3.79, 40),(3.77, 30),(3.74, 20),(3.68, 10),(3.45, 5),(3.0, 0),
    ]
    
    # 遍历电压表
    for i in range(len(voltage_to_percentage) - 1):
        v1, p1 = voltage_to_percentage[i]
        v2, p2 = voltage_to_percentage[i + 1] 
        
        # 如果电压在区间 [v2, v1] 之间
        if v2 <= voltage <= v1:
            # 线性插值
            percentage = p1 + (voltage - v1) * (p2 - p1) / (v2 - v1)
            return round(percentage, 2)  # 返回百分比，保留两位小数
    
    # 如果电压超出范围
    if voltage > 4.2:
        return 100
    elif voltage < 3.0:
        return 0

class DeviceSelectionDialog(QDialog):
    """设备选择对话框"""
    def __init__(self, devices, mode="train"):
        super().__init__()
        self.mode = mode  # 模式：'train'=训练模式，'predict'=预测模式
        
        if mode == "train":
            self.setWindowTitle("选择训练设备和模型")
        else:
            self.setWindowTitle("选择预测模型")
            
        self.setGeometry(400, 300, 300, 300)  # 增加高度以容纳模型选择

        self.selected_devices = []  # 用于存储选中的设备名称
        self.selected_model = "transformer"  # 默认选择transformer模型

        layout = QVBoxLayout()

        # 转换设备列表为字符串并去除空格，确保统一性
        self.devices = [str(device).strip() for device in devices]

        # 根据模式显示不同内容
        if mode == "train":
            # 训练模式 - 显示设备选择
            label = QLabel("请选择需要训练的设备：")
            layout.addWidget(label)

            # 创建复选框
            self.checkboxes = []
            for device in self.devices:
                checkbox = QCheckBox(device)
                self.checkboxes.append(checkbox)
                layout.addWidget(checkbox)  # 添加到布局
        else:
            # 预测模式 - 显示提示
            label = QLabel("请选择用于预测的模型类型：")
            layout.addWidget(label)
            
            # 添加说明
            info_label = QLabel("选择不同的模型类型将影响预测结果。\n"
                               "Transformer: 更适合复杂的时序关系\n"
                               "DLinear: 更适合线性关系和季节性变化")
            info_label.setStyleSheet("color: blue;")
            layout.addWidget(info_label)

        # 添加模型选择部分
        model_group = QGroupBox("选择模型类型")
        model_layout = QVBoxLayout()

        # 创建单选按钮组
        self.model_buttons = QButtonGroup()
        
        # Transformer模型选项
        transformer_radio = QRadioButton("Transformer")
        transformer_radio.setChecked(True)  # 默认选中
        self.model_buttons.addButton(transformer_radio)
        model_layout.addWidget(transformer_radio)
        
        # DLinear模型选项
        dlinear_radio = QRadioButton("DLinear")
        self.model_buttons.addButton(dlinear_radio)
        model_layout.addWidget(dlinear_radio)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 确定按钮
        confirm_button = QPushButton("确定")
        confirm_button.clicked.connect(self.on_confirm)
        layout.addWidget(confirm_button)

        self.setLayout(layout)

    def on_confirm(self):
        """获取选中的设备和模型类型"""
        if self.mode == "train":
            # 训练模式 - 需要选择设备
            self.selected_devices = [cb.text() for cb in self.checkboxes if cb.isChecked()]
            if not self.selected_devices:
                QMessageBox.warning(self, "提示", "未选择任何设备，请选择至少一个设备。")
                return
        else:
            # 预测模式 - 不需要选择设备
            self.selected_devices = []
            
        # 获取选中的模型类型
        selected_button = self.model_buttons.checkedButton()
        if selected_button.text() == "DLinear":
            self.selected_model = "dlinear"
        else:
            self.selected_model = "transformer"
            
        self.accept()  # 关闭对话框


# 弹窗类
class ProgressDialog(QDialog):
    """用于显示操作进度的弹窗"""
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 200)

        # 布局设置
        layout = QVBoxLayout()
        
        # 添加居中的标签
        label = QLabel(message)
        label.setFont(QFont("Arial", 14))
        label.setAlignment(Qt.AlignCenter)  # 设置文字居中
        layout.addWidget(label)

        self.setLayout(layout)

        # 将弹窗居中放置在父窗口中
        if parent:
            self.center_on_parent(parent)

    def center_on_parent(self, parent):
        """将弹窗居中显示在父窗口中"""
        parent_geometry = parent.geometry()
        dialog_geometry = self.geometry()
        x = parent_geometry.x() + (parent_geometry.width() - dialog_geometry.width()) // 2
        y = parent_geometry.y() + (parent_geometry.height() - dialog_geometry.height()) // 2
        self.move(x, y)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("绝缘子设备数据管理系统")
        self.setGeometry(100, 100, 1450, 800)
        self.config = Config(DEVICE_LOCATIONS_CONFIG_PATH)

        # 初始化管理员模式（默认为非管理员模式）
        self.admin_mode = False
        self.admin_password = self.config.get_config("DEFAULT", "admin_password")
        if not self.admin_password or self.admin_password == "NOT FOUND":
            self.admin_password = None  # 未配置密码时禁止进入管理员模式。

        # 安装事件过滤器以捕获全局按键事件
        self.installEventFilter(self)

        # 主布局使用 QTabWidget
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # 初始化配置和设备信息
        self.config = Config(CONFIG_PATH)
        self.devices = self.config.get_config_list("QUERY", "devices")

        # 初始化主界面和设备详情界面
        self.main_tab = QWidget()
        self.main_tab2 = QWidget()  # 新增的主界面2
        self.device_details_tab = QWidget()
        self.tabs.addTab(self.main_tab, "主界面")
        self.tabs.addTab(self.device_details_tab, "绝缘子串详情")
        # 不初始添加主界面2标签页，等待切换到管理员模式再添加

        # 存储当前显示的设备索引
        self.current_device_index = None
        # 记录来源界面
        self.details_source = None  # 'main' 或 'main2'

        # 初始化主界面布局
        self.init_main_tab()
        # 初始化主界面2布局
        self.init_main_tab2()
        # 初始化设备详情布局
        self.init_device_details_tab()

        # 加载设备位置信息
        self.load_device_locations()  # 自动加载已保存的位置信息

        # 初始化数据抓取器和模型训练器
        self.data_fetcher = DataFetcher(
            config_file=CONFIG_PATH,
            output_folder=OUTPUT_DIR,
        )
        self.model_trainer = DeviceModelTrainer(
            config_file=CONFIG_PATH,
            data_folder=OUTPUT_DIR,
        )
        self.ensemble_predictor = None
        manifest_path = os.path.join(
            str(DIST_DIR),
            "humidity_stage_feature_ensemble_evaluation",
            "ensemble_manifest.json",
        )
        if os.path.exists(manifest_path):
            try:
                self.ensemble_predictor = HumidityStageEnsemblePredictor(manifest_path)
                print(f"已加载湿度阶段集成模型: {manifest_path}")
            except Exception as ensemble_error:
                print(f"湿度阶段集成模型加载失败，将使用旧模型: {ensemble_error}")

    def predict_with_stage_ensemble(
        self,
        device_name,
        location,
        records,
        temp,
        hum,
        timestamp,
        actual_measure,
    ):
        if self.ensemble_predictor is None:
            raise ValueError("湿度阶段集成模型未加载")
        result = self.ensemble_predictor.predict_from_records(
            imei=str(device_name),
            location=location,
            records=records,
            target_temp=temp,
            target_hum=hum,
            target_timestamp=timestamp,
            actual_measure=actual_measure,
        )
        print(
            "阶段集成预测: device={} model={} P50={:.4f} interval=[{:.4f}, {:.4f}]".format(
                device_name,
                result["SelectedModel"],
                result["P50"],
                result["P10"],
                result["P90"],
            )
        )
        return self.ensemble_predictor.legacy_tuple(result)

    def set_table_item(self, row, column, text, table=None):
        """设置表格中的指定单元格内容"""
        if table is None:
            table = self.status_table
            
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)  # 设置文本居中
        table.setItem(row, column, item)

    def init_main_tab(self):
        """初始化主界面布局"""
        layout = QVBoxLayout()
        font = QFont("Arial", 16)
        self.exit_button = QPushButton("退出")
        self.exit_button.clicked.connect(self.handle_exit)  # 绑定退出逻辑
        self.label = QLabel("")  # 删除"设备状态: "文本
        self.label.setFont(font)
        layout.addWidget(self.label)

        # 添加设备状态表格
        self.status_table = QTableWidget()
        self.status_table.setColumnCount(9)  # 9列
        self.status_table.setHorizontalHeaderLabels(["绝缘子串", "线路电压", "设备数量", "设备状态", "最新数据时间", "温度", "湿度", "绝缘程度", "绝缘状态"])
        self.status_table.setFont(font)
        layout.addWidget(self.status_table)

        # 初始化控制按钮
        self.control_layout = QHBoxLayout()

        self.predict_button = QPushButton("刷新数据")
        self.predict_button.setFont(font)
        self.predict_button.clicked.connect(self.update_predictions)
        self.control_layout.addWidget(self.predict_button)

        self.exit_button = QPushButton("退出")
        self.exit_button.setFont(font)
        self.exit_button.clicked.connect(self.close)
        self.control_layout.addWidget(self.exit_button)

        layout.addLayout(self.control_layout)

        # 设置主界面布局
        self.main_tab.setLayout(layout)

        # 初始化设备状态表格内容
        self.initialize_table()

    def initialize_table(self):
        """初始化设备状态表格"""
        # 存储绝缘子串信息和对应的设备索引
        string_info_dict = {}  # 格式: {'串1': [device_indexes]}
        
        # 先收集所有设备的绝缘子串信息
        for i, device_name in enumerate(self.devices):
            # 获取位置信息
            location = self.config.get_config("DeviceLocations", str(device_name))
            string_id = ""
            
            if location and location != 'NOT FOUND':
                # 尝试从位置字符串中提取第一个下划线前的内容作为绝缘子串信息
                parts = location.split('_')
                if parts:
                    string_id = parts[0]  # 第一部分通常是绝缘子串信息（如"串1"）
            
            # 如果没有提取到绝缘子串信息，则使用默认值
            if not string_id:
                string_id = f"串{i+1}"
                
            print(f"设备 {device_name} 的绝缘子串信息: {string_id}")
                
            # 将设备添加到对应的绝缘子串组中
            if string_id not in string_info_dict:
                string_info_dict[string_id] = []
            string_info_dict[string_id].append(i)
        
        # 打印调试信息
        print("检测到的绝缘子串分组：")
        for string_id, device_indexes in string_info_dict.items():
            device_names = [self.devices[idx] for idx in device_indexes]
            print(f"{string_id}: {device_names}")
        
        # 按绝缘子串ID排序
        sorted_strings = sorted(string_info_dict.keys())
        
        # 设置表格行数为绝缘子串数量
        self.status_table.setRowCount(len(sorted_strings))
        
        # 设置字体
        font = QFont("Arial", 14)  # 字体 Arial，大小 14
        self.status_table.setFont(font)  # 统一设置表格字体
        
        # 保存绝缘子串到设备的映射关系，用于后续更新数据和跳转到设备详情
        self.line_to_devices = {}  # 格式: {行索引: [设备索引]}
        self.line_names = {}       # 存储行索引到绝缘子串名称的映射
        self.line_voltages = {}    # 存储行索引到线路电压的映射
        
        # 初始化表格内容
        for row, string_id in enumerate(sorted_strings):
            device_indexes = string_info_dict[string_id]
            self.line_to_devices[row] = device_indexes
            self.line_names[row] = string_id  # 保存绝缘子串名称
            
            # 从设备配置信息中提取线路电压
            voltage = "未知"
            for device_idx in device_indexes:
                device_name = self.devices[device_idx]
                location = self.config.get_config("DeviceLocations", str(device_name))
                if location and location != 'NOT FOUND':
                    # 尝试从位置字符串中提取电压信息
                    parts = location.split('_')
                    if len(parts) > 1:
                        voltage = parts[1]  # 第二部分通常是电压
                        break
            
            self.line_voltages[row] = voltage  # 保存线路电压
            
            # 设置绝缘子串信息 - 显示绝缘子串ID
            item = QTableWidgetItem(string_id)  # 显示绝缘子串ID
            item.setData(Qt.UserRole, device_indexes[0])  # 存储第一个设备索引
            
            # 如果这个绝缘子串有多个设备，存储所有设备索引
            if len(device_indexes) > 1:
                item.setData(Qt.UserRole + 1, device_indexes)  # 存储所有设备索引
            
            # 设置为可选择但不可编辑
            item.setFlags(item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)  # 移除可编辑标志
            # 设置文本居中对齐
            item.setTextAlignment(Qt.AlignCenter)
            self.status_table.setItem(row, 0, item)
            
            # 初始化其他列
            for col in range(1, 9):  # 9列
                item = QTableWidgetItem("-")
                self.status_table.setItem(row, col, item)
            
            # 设备数量将在刷新数据时显示，这里不再初始化
        
        # 设置列宽
        self.status_table.setColumnCount(9)  # 9列
        self.status_table.setHorizontalHeaderLabels(["绝缘子串", "线路电压", "设备数量", "设备状态", "最新数据时间", "温度", "湿度", "绝缘程度", "绝缘状态"])
        self.status_table.setColumnWidth(0, 120)  # 绝缘子串列宽度
        self.status_table.setColumnWidth(1, 130)  # 线路电压列宽度
        self.status_table.setColumnWidth(2, 130)  # 设备数量列宽度
        self.status_table.setColumnWidth(3, 130)  # 设备状态列宽度
        self.status_table.setColumnWidth(4, 300)  # 最新数据时间列宽度
        self.status_table.setColumnWidth(5, 120)  # 温度列宽度
        self.status_table.setColumnWidth(6, 120)  # 湿度列宽度
        self.status_table.setColumnWidth(7, 180)  # 绝缘程度列宽度
        self.status_table.setColumnWidth(8, 180)  # 绝缘状态列宽度
        
        # 添加点击事件跳转到设备详情
        self.status_table.itemClicked.connect(self.on_device_name_clicked)

    def init_device_details_tab(self):
        """初始化绝缘子串详情界面"""
        # 创建一个垂直布局来包含所有内容
        layout = QVBoxLayout()

        # 创建一个表单布局来显示设备详情
        form_layout = QFormLayout()

        font = QFont("Arial", 14)  # 设置统一字体
        info_font = QFont("Arial", 14)  # 信息显示的字体

        # 创建设备详情的标签和信息显示
        self.device_location_label = QLabel("位置编号: ")
        self.device_location_info = QLabel("暂无数据")
        self.device_location_label.setFont(font)
        self.device_location_info.setFont(info_font)

        # 删除序号相关的标签和信息显示
        # self.device_number_label = QLabel("序号: ")
        # self.device_number_info = QLabel("暂无数据")
        # self.device_number_label.setFont(font)
        # self.device_number_info.setFont(info_font)

        self.device_name_label = QLabel("绝缘子串ID: ")
        self.device_name_info = QLabel("暂无数据")
        self.device_name_label.setFont(font)
        self.device_name_info.setFont(info_font)

        self.device_time_label = QLabel("最新数据时间: ")
        self.device_time_info = QLabel("暂无数据")
        self.device_time_label.setFont(font)
        self.device_time_info.setFont(info_font)

        self.device_temp_label = QLabel("温度: ")
        self.device_temp_info = QLabel("暂无数据")
        self.device_temp_label.setFont(font)
        self.device_temp_info.setFont(info_font)

        self.device_hum_label = QLabel("湿度: ")
        self.device_hum_info = QLabel("暂无数据")
        self.device_hum_label.setFont(font)
        self.device_hum_info.setFont(info_font)

        self.device_insulation_label = QLabel("绝缘程度: ")
        self.device_insulation_info = QLabel("暂无数据")
        self.device_insulation_label.setFont(font)
        self.device_insulation_info.setFont(info_font)

        self.device_battery_label = QLabel("电池电量: ")
        self.device_battery_info = QLabel("暂无数据")
        self.device_battery_label.setFont(font)
        self.device_battery_info.setFont(info_font)
        
        # 新增线路电压和设备状态显示
        self.device_voltage_label = QLabel("线路电压: ")
        self.device_voltage_info = QLabel("暂无数据")
        self.device_voltage_label.setFont(font)
        self.device_voltage_info.setFont(info_font)
        
        self.device_status_label = QLabel("设备状态: ")
        self.device_status_info = QLabel("暂无数据")
        self.device_status_label.setFont(font)
        self.device_status_info.setFont(info_font)

        # 将设备详情信息添加到表单布局
        form_layout.addRow(self.device_location_label, self.device_location_info)
        # 删除序号行
        # form_layout.addRow(self.device_number_label, self.device_number_info)
        form_layout.addRow(self.device_name_label, self.device_name_info)
        form_layout.addRow(self.device_time_label, self.device_time_info)
        form_layout.addRow(self.device_temp_label, self.device_temp_info)
        form_layout.addRow(self.device_hum_label, self.device_hum_info)
        form_layout.addRow(self.device_insulation_label, self.device_insulation_info)
        form_layout.addRow(self.device_battery_label, self.device_battery_info)
        form_layout.addRow(self.device_voltage_label, self.device_voltage_info)
        form_layout.addRow(self.device_status_label, self.device_status_info)

        # 创建并设置更新按钮
        self.update_button = QPushButton("更新数据")
        self.update_button.setFont(font)
        self.update_button.setFixedSize(200, 60)  # 缩小按钮大小
        self.update_button.clicked.connect(self.update_device_info)

        # 创建一个水平布局将按钮居中
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.update_button)
        button_layout.setAlignment(Qt.AlignCenter)  # 将按钮居中

        # 将表单布局和按钮布局添加到主布局
        layout.addLayout(form_layout)  # 添加设备详情的表单布局
        layout.addLayout(button_layout)  # 添加按钮布局

        # 设置设备详情页的布局
        self.device_details_tab.setLayout(layout)

    def update_device_info(self):
        """更新绝缘子串详情信息"""
        if self.current_device_index is not None:
            device_name = self.devices[self.current_device_index]
            
            # 显示加载中状态
            self.device_time_info.setText("正在加载...")
            self.device_temp_info.setText("正在加载...")
            self.device_hum_info.setText("正在加载...")
            self.device_insulation_info.setText("正在加载...")
            self.device_status_info.setText("正在检测...")
            self.device_status_info.setStyleSheet("QLabel { color: black; }")
            
            if self.details_source == 'main2':  # 只有在显示全部内容时才需要显示电池信息
                self.device_battery_info.setText("正在加载...")
            QApplication.processEvents()  # 更新界面显示

            try:
                latest_data = self.data_fetcher.get_latest_data(device_name)

                if latest_data:
                    # 设备在线，更新状态
                    self.device_status_info.setText("正常")
                    self.device_status_info.setStyleSheet("QLabel { color: black; }")
                    
                    analog, temp, hum, timestamp, battery, version = latest_data
                    self.device_time_info.setText(str(timestamp))
                    self.device_temp_info.setText(f"{temp:.2f} °C")
                    self.device_hum_info.setText(f"{hum:.2f} %")
                    if self.details_source == 'main2':  # 只有在显示全部内容时才需要显示电池信息
                        self.device_battery_info.setText(f"{calculate_battery_percentage(battery)}%")

                    # 获取历史测量值用于预测绝缘程度
                    history_count = (
                        self.ensemble_predictor.required_history_records
                        if self.ensemble_predictor is not None
                        else 2
                    )
                    device_data = self.data_fetcher.get_last_n_data(
                        device_name,
                        history_count,
                    )
                    if device_data and len(device_data) >= 2:
                        analogs = [data[0] for data in device_data]
                        try:
                            if self.ensemble_predictor is not None:
                                try:
                                    location = self.device_locations.get(
                                        str(device_name),
                                        self.config.get_config(
                                            "DeviceLocations",
                                            str(device_name),
                                        ),
                                    )
                                    (
                                        pred,
                                        error,
                                        is_anomaly,
                                        status_description,
                                        confidence,
                                        insulation_percentage,
                                    ) = self.predict_with_stage_ensemble(
                                        device_name,
                                        location,
                                        device_data,
                                        temp,
                                        hum,
                                        timestamp,
                                        analog,
                                    )
                                    if insulation_percentage >= 80:
                                        status_description = "绝缘良好"
                                    elif insulation_percentage >= 50:
                                        status_description = "绝缘正常"
                                    else:
                                        status_description = "绝缘异常"
                                    insulation_text = (
                                        f"{insulation_percentage:.1f} % "
                                        f"({status_description})"
                                    )
                                    self.device_insulation_info.setText(insulation_text)
                                    self.device_insulation_info.setStyleSheet(
                                        "QLabel { background-color: transparent; }"
                                    )
                                    return
                                except Exception as ensemble_error:
                                    print(
                                        f"阶段集成预测失败，回退旧模型: {ensemble_error}"
                                    )

                            # ---------模型选择与主界面一致---------
                            imei_suffix = str(device_name)[-4:]
                            model_files = []
                            # 优先带时序特征模型
                            ts_matches = [f for f in os.listdir(MODEL_OUTPUT_DIR) if f.startswith(f"model_{imei_suffix}_transformer_ts") and f.endswith(".pkl")]
                            if ts_matches:
                                ts_matches.sort(reverse=True)
                                model_files.append(os.path.join(MODEL_OUTPUT_DIR, ts_matches[0]))
                            # 标准Transformer模型
                            standard_model = f"model_{imei_suffix}_transformer.pkl"
                            standard_path = os.path.join(MODEL_OUTPUT_DIR, standard_model)
                            if os.path.exists(standard_path):
                                model_files.append(standard_path)
                            # 旧格式模型
                            old_model = f"model_{imei_suffix}.pkl"
                            old_path = os.path.join(MODEL_OUTPUT_DIR, old_model)
                            if os.path.exists(old_path):
                                model_files.append(old_path)
                            # 加载模型
                            model_loaded = False
                            for model_path in model_files:
                                try:
                                    self.model_trainer.load_model(model_path)
                                    model_loaded = True
                                    break
                                except Exception as e:
                                    continue
                            if not model_loaded:
                                self.device_insulation_info.setText("无可用模型")
                                self.device_insulation_info.setStyleSheet("QLabel { background-color: transparent; }")
                                return
                            # ---------模型选择与主界面一致---------
                            pred, error, is_anomaly, status_description, confidence, insulation_percentage = self.model_trainer.predict(temp, hum, analogs[:2])
                            
                            # 根据新的绝缘程度阈值更新状态描述
                            if insulation_percentage >= 80:
                                status_description = "绝缘良好"
                            elif insulation_percentage >= 50:
                                status_description = "绝缘正常"
                            else:
                                status_description = "绝缘异常"
                            
                            # 设置绝缘程度文本，不添加颜色
                            insulation_text = f"{insulation_percentage:.1f} % ({status_description})"
                            self.device_insulation_info.setText(insulation_text)
                            
                            # 删除颜色样式代码，保持普通显示
                            self.device_insulation_info.setStyleSheet("QLabel { background-color: transparent; }")
                        except Exception as predict_e:
                            self.device_insulation_info.setText("预测失败")
                            self.device_insulation_info.setStyleSheet("QLabel { background-color: transparent; }")
                    else:
                        self.device_insulation_info.setText("历史数据不足")
                        self.device_insulation_info.setStyleSheet("QLabel { background-color: transparent; }")
                else:
                    # 设备离线，更新状态
                    self.device_status_info.setText("离线")
                    self.device_status_info.setStyleSheet("QLabel { color: red; }")
                    
                    self.device_time_info.setText("暂无数据")
                    self.device_temp_info.setText("暂无数据")
                    self.device_hum_info.setText("暂无数据")
                    self.device_insulation_info.setText("暂无数据")
                    if self.details_source == 'main2':
                        self.device_battery_info.setText("暂无数据")
                    
                    QMessageBox.warning(self, "警告", f"无法获取绝缘子串 {device_name} 的最新数据。")
                    self.device_insulation_info.setStyleSheet("QLabel { background-color: transparent; }")
            except Exception as e:
                # 发生异常，设置为离线状态
                self.device_status_info.setText("离线")
                self.device_status_info.setStyleSheet("QLabel { color: red; }")
                
                error_message = str(e)
                if "Failed to get timed status for IMEI" in error_message:
                    QMessageBox.critical(self, "错误", f"无法获取绝缘子串 {device_name} 的状态信息！")
                else:
                    QMessageBox.critical(self, "错误", f"更新绝缘子串信息时发生错误: {e}")
        else:
            QMessageBox.warning(self, "警告", "请先选择绝缘子串。")

    def on_device_name_clicked(self, item):
        """处理设备名称点击事件"""
        if item.column() == 0:  # 点击设备名称列
            if item.tableWidget() == self.status_table:
                # 从主界面点击 - 获取表格行索引
                row = item.row()
                # 如果是合并的行，使用第一个设备的索引
                if row in self.line_to_devices and self.line_to_devices[row]:
                    self.current_device_index = self.line_to_devices[row][0]
                else:
                    # 兼容旧逻辑
                    device_index = item.data(Qt.UserRole)
                    if device_index is not None:
                        self.current_device_index = device_index
                    else:
                        return
                
                self.details_source = 'main'
                # 从主界面表格获取绝缘子串信息
                string_id = item.text()  # 直接使用显示的绝缘子串文本
                if not string_id.strip() and hasattr(self, 'line_names') and row in self.line_names:
                    # 如果表格中绝缘子串为空，但有保存的绝缘子串名称，则使用保存的名称
                    string_id = self.line_names[row]
                
                # 如果依然为空，使用默认绝缘子串名称
                if not string_id.strip():
                    string_id = f"串{row+1}"
                
                # 获取完整的位置信息
                device_name = self.devices[self.current_device_index]
                location = self.device_locations.get(str(device_name), string_id)
            else:  # 点击的是主界面2的表格
                # 验证管理员模式
                if not self.admin_mode:
                    QMessageBox.warning(self, "权限不足", "请先使用管理员模式（Ctrl+Z）访问此功能。")
                    return
                
                self.details_source = 'main2'
                # 从主界面2表格获取设备索引
                device_index = item.data(Qt.UserRole)
                if device_index is None:
                    return
                self.current_device_index = device_index
                
                # 获取设备位置信息
                device_name = self.devices[device_index]
                location = "未知位置"
                
                # 从device_locations获取
                if hasattr(self, 'device_locations'):
                    if str(device_name) in self.device_locations:
                        location = self.device_locations[str(device_name)]

            # 更新绝缘子串详情
            self.update_device_details_tab(self.current_device_index, location)
            
            # 自动更新设备最新数据
            self.update_device_info()

            # 切换到绝缘子串详情选项卡
            self.tabs.setCurrentWidget(self.device_details_tab)

    def update_device_details_tab(self, device_index, location):
        """更新绝缘子串详情内容"""
        device_name = self.devices[device_index]  # 假设 self.devices 是一个列表或字典
        
        # 获取绝缘子串ID
        string_id = self.device_string_ids.get(str(device_name), "")
        if not string_id:
            # 尝试从位置信息中提取
            parts = location.split('_')
            string_id = parts[0] if parts else f"串{device_index+1}"
        
        # 根据来源界面显示不同内容
        if self.details_source == 'main':
            # 从主界面来，显示绝缘子串、温度、湿度、绝缘程度、线路电压和设备状态
            # 修改标签文字，突出显示"绝缘子串"而不是"位置编号"
            self.device_location_label.setText("绝缘子串: ")
            self.device_location_info.setText(string_id)
            
            # 获取线路电压信息
            voltage = "未知"
            if hasattr(self, 'device_voltages') and str(device_name) in self.device_voltages:
                voltage = self.device_voltages[str(device_name)]
            self.device_voltage_info.setText(voltage)
            
            # 隐藏不需要显示的字段
            # 不再设置device_number_label和device_number_info的可见性
            self.device_name_label.setVisible(False)
            self.device_name_info.setVisible(False)
            self.device_battery_label.setVisible(False)
            self.device_battery_info.setVisible(False)
            
            # 确保需要的字段可见
            self.device_location_label.setVisible(True)
            self.device_location_info.setVisible(True)
            self.device_time_label.setVisible(True)
            self.device_time_info.setVisible(True)
            self.device_temp_label.setVisible(True)
            self.device_temp_info.setVisible(True)
            self.device_hum_label.setVisible(True)
            self.device_hum_info.setVisible(True)
            self.device_insulation_label.setVisible(True)
            self.device_insulation_info.setVisible(True)
            self.device_voltage_label.setVisible(True)
            self.device_voltage_info.setVisible(True)
            self.device_status_label.setVisible(True)
            self.device_status_info.setVisible(True)
        else:  # 来自主界面2
            # 显示全部字段
            self.device_location_label.setText("位置信息: ")
            self.device_location_info.setText(location)
            # 不再设置序号信息
            # self.device_number_info.setText(str(device_index + 1))
            self.device_name_info.setText(str(device_name))
            
            # 获取线路电压信息
            voltage = "未知"
            if hasattr(self, 'device_voltages') and str(device_name) in self.device_voltages:
                voltage = self.device_voltages[str(device_name)]
            self.device_voltage_info.setText(voltage)
            
            # 确保所有字段可见
            self.device_location_label.setVisible(True)
            self.device_location_info.setVisible(True)
            # 不再设置序号标签的可见性
            # self.device_number_label.setVisible(True)
            # self.device_number_info.setVisible(True)
            self.device_name_label.setVisible(True)
            self.device_name_info.setVisible(True)
            self.device_time_label.setVisible(True)
            self.device_time_info.setVisible(True)
            self.device_temp_label.setVisible(True)
            self.device_temp_info.setVisible(True)
            self.device_hum_label.setVisible(True)
            self.device_hum_info.setVisible(True)
            self.device_insulation_label.setVisible(True)
            self.device_insulation_info.setVisible(True)
            self.device_battery_label.setVisible(True)
            self.device_battery_info.setVisible(True)
            self.device_voltage_label.setVisible(True)
            self.device_voltage_info.setVisible(True)
            self.device_status_label.setVisible(True)
            self.device_status_info.setVisible(True)
        
        # 设置初始值
        self.device_time_info.setText("暂无数据")
        self.device_temp_info.setText("暂无数据")
        self.device_hum_info.setText("暂无数据")
        self.device_insulation_info.setText("暂无数据")
        self.device_status_info.setText("离线")
        self.device_status_info.setStyleSheet("QLabel { color: red; }")
        
        if self.details_source == 'main2':  # 只有在显示全部内容时才需要设置
            self.device_battery_info.setText("暂无数据")
            
        # 清除之前的颜色样式
        self.device_insulation_info.setStyleSheet("QLabel { background-color: transparent; }")

    def update_predictions(self):
        """刷新数据并弹出选择操作的弹窗"""
        # 创建弹窗
        dialog = QDialog(self)
        dialog.setWindowTitle("选择操作")
        dialog.setModal(True)
        dialog.resize(400, 200)  # 增加高度以容纳更多内容

        # 布局
        layout = QVBoxLayout()

        # 添加提示文字
        if self.admin_mode:
            # 管理员模式下显示选择操作提示
            label = QLabel("请选择要进行的操作：")
            label.setFont(QFont("Arial", 16))
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            
            # 管理员模式下显示校准提示
            warning_label = QLabel("注意：初次使用软件请先校准，否则无法预测。\n若长期未进行设备校准预测结果误差较大,请按需校准!")
            warning_font = QFont("Arial", 12)
            warning_font.setWeight(QFont.Bold)  # 设置加粗
            warning_label.setFont(warning_font)
            warning_label.setStyleSheet("color: #FF4500;")  # 设置字体颜色为红橙色
            warning_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(warning_label)
            
            admin_label = QLabel("当前为管理员模式，可以选择指定设备进行校准")
            admin_label.setFont(QFont("Arial", 10))
            admin_label.setStyleSheet("color: blue;")
            admin_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(admin_label)
        else:
            # 普通用户模式下只显示刷新提示
            refresh_label = QLabel("是否刷新数据？")
            refresh_font = QFont("Arial", 12)
            refresh_font.setWeight(QFont.Bold)  # 设置加粗
            refresh_label.setFont(refresh_font)
            refresh_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(refresh_label)
        
        # 按钮布局
        button_layout = QHBoxLayout()

        # 管理员模式下才显示校准按钮
        if self.admin_mode:
            # 数据预测校准按钮
            calibrate_button = QPushButton("预测校准")
            calibrate_button.setFont(QFont("Arial", 12))
            calibrate_button.clicked.connect(lambda: self.show_calibration_dialog(dialog))
            button_layout.addWidget(calibrate_button)
            
            # 模型选择按钮
            model_select_button = QPushButton("选择模型")
            model_select_button.setFont(QFont("Arial", 12))
            model_select_button.clicked.connect(lambda: self.show_model_selection_dialog(dialog))
            button_layout.addWidget(model_select_button)

        # 预测刷新按钮
        refresh_button = QPushButton("确认刷新")
        refresh_button.setFont(QFont("Arial", 12))
        refresh_button.clicked.connect(lambda: self.refresh_data(dialog))
        button_layout.addWidget(refresh_button)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec_()  # 显示弹窗
        
    def show_model_selection_dialog(self, parent_dialog):
        """显示模型选择对话框"""
        parent_dialog.accept()  # 关闭父对话框
        
        # 显示模型选择对话框
        model_dialog = DeviceSelectionDialog(self.devices, mode="predict")
        if model_dialog.exec_() != QDialog.Accepted:
            return
            
        # 保存选择的模型类型
        self.selected_model = model_dialog.selected_model
        
        # 显示选择结果
        QMessageBox.information(
            self, 
            "模型选择", 
            f"已选择 {self.selected_model.upper()} 模型作为默认预测模型。\n"
            f"此设置将在下次刷新数据或校准时生效。"
        )

    def show_calibration_dialog(self, parent_dialog):
        """显示模型校准对话框 - 普通权限与管理员权限区分"""
        parent_dialog.accept()  # 关闭父对话框
        
        # 根据管理员模式区分功能
        if not self.admin_mode:
            # 普通用户模式 - 直接校准所有设备
            confirm = QMessageBox.question(
                self, 
                "确认校准", 
                "过程可能需要较长时间。确定继续吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if confirm == QMessageBox.Yes:
                # 使用所有设备进行校准 - 确保转换为字符串列表
                all_devices = [str(device) for device in self.devices] if self.devices else []
                
                if not all_devices:
                    QMessageBox.warning(self, "警告", "没有找到任何设备，无法进行校准。")
                    return
                    
                print(f"开始校准所有设备: {len(all_devices)} 个设备")
                self.calibrate_predictions(all_devices)
        else:
            # 管理员模式 - 显示设备选择对话框
            device_dialog = DeviceSelectionDialog(self.devices, mode="train")
            if device_dialog.exec_() != QDialog.Accepted:
                return
            
            selected_devices = device_dialog.selected_devices
            if not selected_devices:
                QMessageBox.warning(self, "警告", "请至少选择一个设备！")
                return
            
            # 保存选择的模型类型
            self.selected_model = device_dialog.selected_model
            print(f"选择了{self.selected_model}模型进行校准")
            
            # 调用校准方法
            self.calibrate_predictions(selected_devices)

    def refresh_data(self, dialog):
        """刷新数据并进行预测"""
        dialog.accept()  # 关闭对话框
        
        # 显示数据获取进度弹窗
        fetch_dialog = QProgressDialog("正在获取最新数据...", "取消", 0, 100, self)
        fetch_dialog.resize(300, 100)
        fetch_dialog.setWindowModality(Qt.WindowModal)
        fetch_dialog.setMinimumDuration(0)
        fetch_dialog.setValue(0)
        fetch_dialog.show()
        QApplication.processEvents()
        
        # 直接获取所有设备的最新数据
        try:
            # 获取所有设备的最新数据
            latest_data_list = self.data_fetcher.get_all_latest_data()
            fetch_dialog.setValue(100)
            fetch_dialog.close()
        except Exception as e:
            fetch_dialog.close()
            error_message = str(e)
            # 检查是否包含"Failed to get timed status for IMEI"错误消息
            if "Failed to get timed status for IMEI" in error_message:
                # 提取IMEI号码
                import re
                imei_match = re.search(r'IMEI (\d+)', error_message)
                imei = imei_match.group(1) if imei_match else "未知"
                QMessageBox.critical(self, "错误", f"无法获取设备 {imei} 的状态信息，刷新中止！")
            else:
                QMessageBox.critical(self, "错误", f"获取最新数据失败: {e}")
            return
        
        # 显示预测进度弹窗
        progress_dialog = QProgressDialog("正在刷新预测数据...", "取消", 0, len(self.devices), self)
        progress_dialog.resize(300, 100)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()

        QApplication.processEvents()

        # 临时存储: 为主界面按行索引存储聚合数据
        aggregated_data = {}  # {row_index: {'temps': [], 'hums': [], 'insulations': [], 'timestamps': [], 'status': [], 'voltages': []}}
        
        # 记录已处理的设备和行的映射
        processed_devices = {}
        # 存储每个设备的状态
        device_status = {}
        
        try:
            # 更新主界面2表格中的设备显示和线路电压信息
            for i, device_name in enumerate(self.devices):
                # 获取绝缘子串ID并更新主界面2表格的设备ID列
                device_id_item = self.status_table2.item(i, 0)
                if device_id_item:
                    # 获取绝缘子串ID
                    string_id = self.device_string_ids.get(str(device_name), "")
                    if not string_id:
                        # 尝试从位置信息中提取
                        location = self.device_locations.get(str(device_name), "")
                        parts = location.split('_')
                        string_id = parts[0] if parts else str(device_name)
                    
                    # 更新设备ID显示为绝缘子串ID + 设备ID
                    display_text = f"{string_id}: {device_name}"
                    device_id_item.setText(display_text)
                
                # 获取该设备的线路电压并更新到主界面2表格
                voltage = self.device_voltages.get(str(device_name), "未知")
                voltage_item = self.status_table2.item(i, 1)  # 线路电压在第2列 (索引1)
                if voltage_item:
                    voltage_item.setText(voltage)
            
            # 先处理所有设备的数据
            for i, latest_data in enumerate(latest_data_list):
                device_name = self.devices[i] if i < len(self.devices) else f"未知设备({i})"
                if not latest_data:
                    # 设置设备状态为离线
                    device_status[i] = "离线"
                    
                    # 更新主界面2的设备状态列
                    status_item = self.status_table2.item(i, 2)  # 设备状态移至第3列 (索引2)
                    if status_item:
                        status_item.setText("离线")
                        status_item.setForeground(QColor(255, 0, 0))  # 红色文字
                        status_item.setBackground(QColor(255, 255, 255))  # 白色背景
                    continue
                
                try:
                    # 设置设备状态为在线
                    device_status[i] = "正常"
                    
                    # 更新主界面2的设备状态列
                    status_item = self.status_table2.item(i, 2)  # 设备状态移至第3列 (索引2)
                    if status_item:
                        status_item.setText("正常")
                        status_item.setForeground(QColor(0, 128, 0))  # 绿色文字
                        status_item.setBackground(QColor(255, 255, 255))  # 白色背景
                    
                    # 解析最新数据
                    analog, temp, hum, timestamp, battery, version = latest_data
                    
                    # 获取设备IMEI号后四位
                    device_name = self.devices[i] if i < len(self.devices) else f"设备{i+1}"
                    imei_suffix = str(device_name)[-4:] if len(str(device_name)) >= 4 else str(device_name)
                    
                    # 获取线路电压信息
                    voltage = self.device_voltages.get(str(device_name), "未知")
                    
                    # 构建模型文件名和路径，根据用户选择的模型类型优先排序
                    model_files = []
                    ensemble_model_marker = "<humidity_stage_ensemble>"
                    if self.ensemble_predictor is not None:
                        model_files.append(ensemble_model_marker)
                        print("优先使用无Battery湿度阶段FMLP集成模型")

                    model_output_files = (
                        os.listdir(MODEL_OUTPUT_DIR)
                        if os.path.isdir(MODEL_OUTPUT_DIR)
                        else []
                    )
                    
                    # 获取用户选择的模型类型，如果没有设置则使用默认值
                    preferred_model_type = getattr(self, 'selected_model', 'transformer')
                    print(f"使用{preferred_model_type}模型进行预测")
                    
                    # 根据用户选择的模型类型优先排序
                    if preferred_model_type == "dlinear":
                        # 优先使用DLinear模型
                        dlinear_model = f"model_{imei_suffix}_dlinear.pkl"
                        dlinear_path = os.path.join(MODEL_OUTPUT_DIR, dlinear_model)
                        if os.path.exists(dlinear_path):
                            model_files.append(dlinear_path)
                        
                        # 然后尝试Transformer模型作为备选
                        ts_pattern = f"model_{imei_suffix}_transformer_ts*.pkl"
                        ts_matches = [f for f in model_output_files if f.startswith(f"model_{imei_suffix}_transformer_ts") and f.endswith(".pkl")]
                        if ts_matches:
                            ts_matches.sort(reverse=True)
                            model_files.append(os.path.join(MODEL_OUTPUT_DIR, ts_matches[0]))
                        
                        standard_model = f"model_{imei_suffix}_transformer.pkl"
                        standard_path = os.path.join(MODEL_OUTPUT_DIR, standard_model)
                        if os.path.exists(standard_path):
                            model_files.append(standard_path)
                    else:
                        # 优先使用Transformer模型
                        # 先尝试匹配带时序特征的Transformer模型
                        ts_pattern = f"model_{imei_suffix}_transformer_ts*.pkl"
                        ts_matches = [f for f in model_output_files if f.startswith(f"model_{imei_suffix}_transformer_ts") and f.endswith(".pkl")]
                        if ts_matches:
                            ts_matches.sort(reverse=True)  # 按文件名排序，通常更高的数字表示更多特征
                            model_files.append(os.path.join(MODEL_OUTPUT_DIR, ts_matches[0]))
                        
                        # 添加标准的Transformer模型
                        standard_model = f"model_{imei_suffix}_transformer.pkl"
                        standard_path = os.path.join(MODEL_OUTPUT_DIR, standard_model)
                        if os.path.exists(standard_path):
                            model_files.append(standard_path)
                            
                        # 然后尝试DLinear模型作为备选
                        dlinear_model = f"model_{imei_suffix}_dlinear.pkl"
                        dlinear_path = os.path.join(MODEL_OUTPUT_DIR, dlinear_model)
                        if os.path.exists(dlinear_path):
                            model_files.append(dlinear_path)
                    
                    # 添加旧格式模型作为最后备选
                    old_model = f"model_{imei_suffix}.pkl"
                    old_path = os.path.join(MODEL_OUTPUT_DIR, old_model)
                    if os.path.exists(old_path):
                        model_files.append(old_path)
                    
                    # 检查是否找到了任何模型
                    if not model_files:
                        QMessageBox.warning(self, "警告", f"设备 {device_name} 没有对应的模型文件。")
                        continue
                    
                    # 尝试按优先级加载模型
                    model_loaded = False
                    for model_path in model_files:
                        try:
                            if model_path == ensemble_model_marker:
                                model_loaded = True
                                print("湿度阶段FMLP集成模型已就绪")
                                break
                            print(f"尝试加载模型: {model_path}")
                            self.model_trainer.load_model(model_path)
                            model_loaded = True
                            print(f"成功加载模型: {model_path}")
                            break  # 成功加载后退出循环
                        except Exception as e:
                            print(f"加载模型 {model_path} 失败: {e}")
                            continue
                    
                    if not model_loaded:
                        QMessageBox.critical(self, "错误", f"设备 {device_name} 的所有可用模型均无法加载。")
                        continue
                    
                    # 进行预测（模型需要历史数据，保留获取历史值的部分）
                    try:
                        # 获取历史测量值用于预测
                        history_count = (
                            self.ensemble_predictor.required_history_records
                            if model_path == ensemble_model_marker
                            else 2
                        )
                        device_data = self.data_fetcher.get_last_n_data(
                            device_name,
                            history_count,
                        )
                        if not device_data or len(device_data) < 2:
                            QMessageBox.warning(self, "警告", f"设备 {device_name} 的历史数据不足，无法进行准确预测。")
                            continue
                            
                            
                        analogs = [data[0] for data in device_data]  # 提取测量值
                        legacy_analogs = analogs[:2]
                        
                        # 使用模型预测
                        try:
                            if model_path == ensemble_model_marker:
                                location = self.device_locations.get(
                                    str(device_name),
                                    self.config.get_config(
                                        "DeviceLocations",
                                        str(device_name),
                                    ),
                                )
                                (
                                    pred,
                                    error,
                                    is_anomaly,
                                    status_description,
                                    confidence,
                                    insulation_percentage,
                                ) = self.predict_with_stage_ensemble(
                                    device_name,
                                    location,
                                    device_data,
                                    temp,
                                    hum,
                                    timestamp,
                                    analog,
                                )
                            else:
                                model_type = self.model_trainer.model_type
                                print(f"使用{model_type}模型进行预测")
                                (
                                    pred,
                                    error,
                                    is_anomaly,
                                    status_description,
                                    confidence,
                                    insulation_percentage,
                                ) = self.model_trainer.predict(
                                    temp,
                                    hum,
                                    legacy_analogs,
                                )
                        except Exception as predict_e:
                            # 如果预测失败，记录错误并尝试使用其他模型
                            print(f"预测失败: {predict_e}")
                            
                            # 尝试继续使用其他模型
                            continue_prediction = False
                            for backup_path in model_files[model_files.index(model_path)+1:]:
                                try:
                                    print(f"尝试使用备用模型: {backup_path}")
                                    self.model_trainer.load_model(backup_path)
                                    pred, error, is_anomaly, status_description, confidence, insulation_percentage = self.model_trainer.predict(temp, hum, legacy_analogs)
                                    print(f"使用备用模型 {backup_path} 预测成功")
                                    continue_prediction = True
                                    break
                                except Exception as backup_e:
                                    print(f"备用模型 {backup_path} 也无法预测: {backup_e}")
                                    continue
                                    
                            if not continue_prediction:
                                raise ValueError(f"所有模型均无法进行预测: {predict_e}")

                        # 根据新的绝缘程度阈值设置颜色
                        if insulation_percentage >= 80:
                            color = QColor(0, 180, 0)  # 深绿色
                        elif insulation_percentage >= 50:
                            color = QColor(144, 238, 144)  # 浅绿色
                        else:
                            color = QColor(255, 0, 0)  # 红色
                        
                        # 根据新的绝缘程度阈值更新状态描述
                        if insulation_percentage >= 80:
                            status_description = "绝缘良好"
                        elif insulation_percentage >= 50:
                            status_description = "绝缘正常"
                        else:
                            status_description = "绝缘异常"
                        
                        # 更新主界面表格 - 添加最新数据时间列
                        self.set_table_item(i, 3, str(timestamp), self.status_table)  # 最新数据时间在第4列 (索引3)
                        self.set_table_item(i, 4, f"{temp:.2f} °C", self.status_table)  # 温度在第5列 (索引4)
                        self.set_table_item(i, 5, f"{hum:.2f} %", self.status_table)  # 湿度在第6列 (索引5)
                        
                        # 创建主界面中的绝缘程度信息
                        insulation_text = f"{insulation_percentage:.1f} %"
                        insulation_item = QTableWidgetItem(insulation_text)
                        insulation_item.setTextAlignment(Qt.AlignCenter)
                        insulation_item.setFont(QFont("Arial", 14))
                        insulation_item.setBackground(color)
                        self.status_table.setItem(i, 6, insulation_item)  # 绝缘程度在第7列 (索引6)
                        
                        # 更新主界面绝缘状态
                        status_item = QTableWidgetItem(status_description)
                        status_item.setTextAlignment(Qt.AlignCenter)
                        status_item.setFont(QFont("Arial", 14))
                        status_item.setBackground(color)
                        self.status_table.setItem(i, 7, status_item)  # 绝缘状态在第8列 (索引7)
                        
                        # 同时更新主界面2的相应列
                        # 先前已更新了设备ID、线路电压、设备状态列，现在更新其他列
                        self.set_table_item(i, 3, str(timestamp), self.status_table2)  # 最新数据时间在第4列 (索引3)
                        self.set_table_item(i, 4, f"{temp:.2f} °C", self.status_table2)  # 温度在第5列 (索引4)
                        self.set_table_item(i, 5, f"{hum:.2f} %", self.status_table2)  # 湿度在第6列 (索引5)
                        
                        # 创建主界面2中的绝缘程度信息
                        insulation_item2 = QTableWidgetItem(insulation_text)
                        insulation_item2.setTextAlignment(Qt.AlignCenter)
                        insulation_item2.setFont(QFont("Arial", 14))
                        insulation_item2.setBackground(color)
                        self.status_table2.setItem(i, 6, insulation_item2)  # 绝缘程度在第7列 (索引6)
                        
                        # 更新主界面2的绝缘状态
                        status_item2 = QTableWidgetItem(status_description)
                        status_item2.setTextAlignment(Qt.AlignCenter)
                        status_item2.setFont(QFont("Arial", 14))
                        status_item2.setBackground(color)
                        self.status_table2.setItem(i, 7, status_item2)  # 绝缘状态在第8列 (索引7)
                        
                        # 找出该设备对应的主界面行
                        processed = False
                        for row, device_indexes in self.line_to_devices.items():
                            if i in device_indexes:
                                # 收集该行的数据，稍后计算平均值
                                if row not in aggregated_data:
                                    aggregated_data[row] = {
                                        'temps': [], 'hums': [], 'insulations': [], 
                                        'timestamps': [], 'colors': [], 'status': [],
                                        'voltages': [], 'device_status': []
                                    }
                                aggregated_data[row]['temps'].append(temp)
                                aggregated_data[row]['hums'].append(hum)
                                aggregated_data[row]['insulations'].append(insulation_percentage)
                                aggregated_data[row]['timestamps'].append(timestamp)
                                aggregated_data[row]['colors'].append(color)
                                aggregated_data[row]['status'].append(status_description)
                                aggregated_data[row]['voltages'].append(voltage)
                                aggregated_data[row]['device_status'].append("正常")
                                
                                processed_devices[i] = row  # 记录设备已被处理到哪一行
                                processed = True
                                break
                        
                        if not processed:
                            print(f"警告: 设备 {device_name} (索引 {i}) 未找到对应的主界面行")

                    except Exception as e:
                        print(f"设备 {device_name} 处理异常: {e}")
                        continue  # 继续处理下一个设备，不中断所有设备的刷新
                except Exception as parse_error:
                    print(f"解析设备 {i} 数据异常: {parse_error}")
                    continue  # 继续处理下一个设备

                # 更新进度条
                progress_dialog.setValue(i + 1)

                # 如果用户点击了取消按钮，则中止操作
                if progress_dialog.wasCanceled():
                    break
            
            # 调试打印
            print("\n处理后的设备分组:")
            for row, data in aggregated_data.items():
                devices_in_row = [i for i, r in processed_devices.items() if r == row]
                device_names = [self.devices[i] if i < len(self.devices) else f"设备{i+1}" for i in devices_in_row]
                line_name = self.line_names.get(row, f"行{row}")  # 获取对应的线路名称，如果没有则显示行号
                print(f"行 {row} ({line_name}): {device_names} - 温度: {data['temps']}, 湿度: {data['hums']}, 绝缘程度: {data['insulations']}")
            
            # 更新主界面表格 - 计算平均值并聚合显示
            for row, data in aggregated_data.items():
                if not data['temps']:  # 没有数据则跳过
                    continue
                
                # 首先确保正确的线路名称显示在第一列
                line_name = self.line_names.get(row, f"线路{row+1}")  # 从保存的映射中获取正确的线路名称
                # 更新线路名称单元格（因为可能存在重置表格的情况）
                line_item = QTableWidgetItem(line_name)
                device_indexes = self.line_to_devices.get(row, [])
                if device_indexes:
                    line_item.setData(Qt.UserRole, device_indexes[0])  # 存储第一个设备索引
                    if len(device_indexes) > 1:
                        line_item.setData(Qt.UserRole + 1, device_indexes)  # 存储所有设备索引
                line_item.setFlags(line_item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)  # 移除可编辑标志
                # 设置文本居中对齐
                line_item.setTextAlignment(Qt.AlignCenter)
                self.status_table.setItem(row, 0, line_item)
                
                # 计算平均值
                avg_temp = sum(data['temps']) / len(data['temps'])
                avg_hum = sum(data['hums']) / len(data['hums'])
                avg_insulation = sum(data['insulations']) / len(data['insulations'])
                
                # 获取线路电压 - 使用第一个设备的电压信息
                voltage = data['voltages'][0] if data['voltages'] else "未知"
                
                # 确定设备状态 - 如果有一个设备在线，显示"正常"，否则显示"离线"
                device_status_text = "正常" if "正常" in data['device_status'] else "离线"
                device_status_color = QColor(200, 255, 200) if device_status_text == "正常" else QColor(255, 200, 200)
                
                # 获取最新的时间戳 - 使用最新的那个
                latest_timestamp = max(data['timestamps']) if data['timestamps'] else "-"
                
                # 更新主界面表格 - 按正确的顺序设置各列数据
                self.set_table_item(row, 1, voltage, self.status_table)  # 线路电压在第2列 (索引1)
                
                # 设置设备状态
                device_status_item = QTableWidgetItem(device_status_text)
                device_status_item.setTextAlignment(Qt.AlignCenter)
                device_status_item.setFont(QFont("Arial", 14))
                device_status_item.setBackground(device_status_color)
                # 使用文本颜色而不是背景色
                if device_status_text == "正常":
                    device_status_item.setForeground(QColor(0, 0, 0))  # 正常状态黑色文字
                else:
                    device_status_item.setForeground(QColor(255, 0, 0))  # 异常状态红色文字
                device_status_item.setBackground(QColor(255, 255, 255))  # 白色背景
                self.status_table.setItem(row, 3, device_status_item)  # 设备状态在第4列 (索引3)
                
                # 设置最新数据时间
                self.set_table_item(row, 4, str(latest_timestamp), self.status_table)  # 最新数据时间在第5列 (索引4)
                
                # 更新其他列
                self.set_table_item(row, 5, f"{avg_temp:.2f} °C", self.status_table)  # 温度在第6列 (索引5) 
                self.set_table_item(row, 6, f"{avg_hum:.2f} %", self.status_table)  # 湿度在第7列 (索引6)
                
                # 根据平均绝缘程度确定颜色和状态
                if avg_insulation >= 80:
                    color = QColor(0, 180, 0)  # 深绿色
                    status_description = "绝缘良好"
                elif avg_insulation >= 50:
                    color = QColor(144, 238, 144)  # 浅绿色
                    status_description = "绝缘正常"
                else:
                    color = QColor(255, 0, 0)  # 红色
                    status_description = "绝缘异常"
                
                # 创建主界面中绝缘程度信息
                insulation_text = f"{avg_insulation:.1f} %"
                insulation_item = QTableWidgetItem(insulation_text)
                insulation_item.setTextAlignment(Qt.AlignCenter)
                insulation_item.setFont(QFont("Arial", 14))
                insulation_item.setBackground(color)
                self.status_table.setItem(row, 7, insulation_item)  # 绝缘程度在第8列 (索引7)
                
                # 更新主界面绝缘状态
                status_item = QTableWidgetItem(status_description)
                status_item.setTextAlignment(Qt.AlignCenter)
                status_item.setFont(QFont("Arial", 14))
                status_item.setBackground(color)
                self.status_table.setItem(row, 8, status_item)  # 绝缘状态在第9列 (索引8)

                # 更新设备数量
                device_count = len(self.line_to_devices[row]) if row in self.line_to_devices else 0
                count_item = QTableWidgetItem(str(device_count))
                count_item.setTextAlignment(Qt.AlignCenter)
                count_item.setFont(QFont("Arial", 14))
                self.status_table.setItem(row, 2, count_item)  # 设备数量在第3列 (索引2)

            # 关闭进度弹窗
            progress_dialog.close()
            QMessageBox.information(self, "完成", "数据刷新完成!")

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(self, "错误", f"更新数据时发生错误: {e}")

    def calibrate_predictions(self, selected_devices):
        """数据预测校准功能"""
        # 确保所有设备都转换为字符串格式
        selected_devices = [str(device) for device in selected_devices]
        
        # 根据设备数量给出更明确的进度提示
        is_all_devices = len(selected_devices) == len(self.devices)
        progress_title = "全设备校准中" if is_all_devices else "预测校准中"
        
        # 显示进度弹窗
        progress_dialog = QProgressDialog("准备开始预测校准...", "取消", 0, 100, self)
        # 设置进度弹窗大小
        progress_dialog.resize(400, 100)
        progress_dialog.setWindowTitle(progress_title)  # 设置弹窗标题
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示进度弹窗
        progress_dialog.setAutoClose(True)  # 完成时自动关闭
        progress_dialog.setAutoReset(False)  # 防止进度条重置
        
        # 获取取消按钮并保存引用
        cancel_button = progress_dialog.findChild(QPushButton)
        
        progress_dialog.show()

        def update_progress(value, message):
            """更新进度条和消息的回调函数"""
            progress_dialog.setLabelText(message)
            progress_dialog.setValue(value)
            
            # 当进度达到100%时，修改按钮文本为"完成"
            if value >= 100 and cancel_button:
                cancel_button.setText("完成")
            
            QApplication.processEvents()  # 确保UI更新

        try:
            model_dir = MODEL_OUTPUT_DIR
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)  # 创建 model_output 文件夹

            # 确保devices_str是字符串列表
            devices_str = [str(device) for device in self.devices]
            
            # 获取最新的数据
            update_progress(5, "正在获取最新数据...")
            try:
                self.data_fetcher.fetch_data()  # 获取最新的数据
            except Exception as e:
                error_message = str(e)
                if "Failed to get timed status for IMEI" in error_message:
                    # 提取IMEI号码
                    import re
                    imei_match = re.search(r'IMEI (\d+)', error_message)
                    imei = imei_match.group(1) if imei_match else "未知"
                    progress_dialog.close()
                    QMessageBox.critical(self, "错误", f"无法获取设备 {imei} 的状态信息，校准中止！")
                    return
                elif "登录失败" in error_message:
                    progress_dialog.close()
                    QMessageBox.critical(self, "错误", "无法连接到服务器，请检查网络连接！")
                    return
                else:
                    progress_dialog.close()
                    QMessageBox.critical(self, "错误", f"获取数据失败: {e}")
                    return
            
            # 更新进度信息根据是否为全设备校准
            if is_all_devices:
                update_progress(10, "已获取数据，开始执行全设备校准...")
            
            total_devices = len(selected_devices)
            for i, device_name in enumerate(selected_devices):
                if progress_dialog.wasCanceled():
                    break
                
                device_str = str(device_name)  # 确保是字符串
                
                # 检查设备是否在列表中
                if device_str not in devices_str:
                    # 尝试查找部分匹配的设备
                    matching_devices = [d for d in devices_str if device_str in d or d in device_str]
                    if matching_devices:
                        device_str = matching_devices[0]  # 使用第一个匹配的设备
                        print(f"找到匹配设备: '{device_str}' 替代 '{device_name}'")
                    else:
                        progress_dialog.close()
                        QMessageBox.critical(self, "错误", f"设备号 '{device_str}' 不在设备列表中！")
                        return

                # 获取设备索引
                try:
                    device_index = devices_str.index(device_str)
                except ValueError:
                    progress_dialog.close()
                    QMessageBox.critical(self, "错误", f"无法确定设备 '{device_str}' 的索引！")
                    return
                
                # 获取设备IMEI的后四位作为文件名的一部分
                imei = device_str
                imei_suffix = imei[-4:] if len(imei) >= 4 else imei

                try:
                    # 执行数据抓取和模型校准，传入模型类型和进度回调
                    print(f"正在训练设备 {device_str} 的模型...")
                    
                    # 创建设备特定的进度回调
                    def device_progress_callback(value, message):
                        # 将每个设备的进度映射到总体进度的对应部分
                        device_progress = 10 + (value * 0.9)  # 将0-100映射到10-100
                        overall_progress = int(device_progress / total_devices + (i * 90 / total_devices))
                        
                        # 如果消息中包含Epoch，替换为简化的消息
                        if "Epoch" in message:
                            display_message = "正在训练模型..."
                        else:
                            display_message = message
                        
                        # 根据是否为全设备校准显示不同信息    
                        if is_all_devices:
                            device_num = i + 1
                            progress_msg = f"设备 {device_num}/{total_devices}: {display_message}"
                        else:
                            progress_msg = f"设备 {device_str}: {display_message}"
                        
                        update_progress(overall_progress, progress_msg)
                    
                    # 获取选择的模型类型
                    model_type = getattr(self, 'selected_model', 'transformer')  # 默认使用transformer
                    
                    self.model_trainer.train_model(
                        device_index, 
                        model_type=model_type,
                        progress_callback=device_progress_callback
                    )

                    # 构建带有描述性信息的模型文件名
                    model_name = f"model_{imei_suffix}_{model_type}"
                    if model_type == "transformer" and hasattr(self.model_trainer.model, 'input_dim'):
                        input_dim = self.model_trainer.model.input_dim
                        if input_dim > 3:
                            model_name += f"_ts{input_dim}"
                    
                    # 保存校准后的模型
                    model_path = os.path.join(model_dir, f"{model_name}.pkl")
                    self.model_trainer.save_model(model_path)
                    print(f"设备 {device_str} 的模型已保存到 {model_path}")
                    
                    # 确认已保存的模型能够被正确加载
                    try:
                        test_trainer = DeviceModelTrainer(
                            config_file=CONFIG_PATH,
                            data_folder=OUTPUT_DIR,
                        )
                        test_trainer.load_model(model_path)
                        print(f"模型验证成功：可以正确加载 {model_path}")
                    except Exception as load_e:
                        print(f"警告: 模型加载验证失败: {load_e}")
                except Exception as train_e:
                    progress_dialog.close()
                    error_msg = str(train_e)
                    if "没有湿度<70%的有效数据点" in error_msg:
                        QMessageBox.critical(self, "错误", f"设备 {device_str} 没有湿度小于70%的有效数据，无法训练模型。\n请确保设备数据中有足够的低湿度环境下的测量值。")
                    else:
                        QMessageBox.critical(self, "错误", f"设备 {device_str} 的模型训练失败: {train_e}")
                    return

            if not progress_dialog.wasCanceled():
                progress_dialog.setValue(100)
                # 确保按钮文本更新为"完成"
                cancel_button = progress_dialog.findChild(QPushButton)
                if cancel_button:
                    cancel_button.setText("完成")
                
                # 根据设备数量提供不同的完成信息
                if is_all_devices:
                    complete_msg = "全部设备校准完成！\n新模型已包含时序特征分析，可能会提供更准确的预测。"
                else:
                    complete_msg = "所选设备校准完成！\n新模型已包含时序特征分析，可能会提供更准确的预测。"
                    
                QMessageBox.information(self, "完成", complete_msg)
        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(self, "错误", f"数据预测校准失败: {e}")

    def load_device_locations(self):
        """从配置文件加载设备位置信息（只读）但不显示在界面中，等待刷新后显示"""
        # 不再在主界面表格显示位置信息，只保存在对象中，等待刷新后显示
        self.device_locations = {}
        self.device_voltages = {}  # 存储每个设备对应的线路电压
        self.device_string_ids = {}  # 存储每个设备对应的绝缘子串ID
        
        # 只保存位置信息以供后续使用
        for i, device_id in enumerate(self.devices):
            # 从配置文件获取位置信息
            location = self.config.get_config("DeviceLocations", str(device_id))
            if location and location != 'NOT FOUND':
                # 解析配置格式: 串1_110kV_第三~四片
                parts = location.split('_')
                
                # 提取绝缘子串ID
                string_id = parts[0] if parts else f"串{i+1}"
                
                # 提取电压信息
                voltage = parts[1] if len(parts) > 1 else "未知"
                
                self.device_locations[str(device_id)] = location
                self.device_voltages[str(device_id)] = voltage
                self.device_string_ids[str(device_id)] = string_id
            else:
                self.device_locations[str(device_id)] = f"串{i+1}"
                self.device_voltages[str(device_id)] = "未知"
                self.device_string_ids[str(device_id)] = f"串{i+1}"

    def save_device_locations(self):
        """保存设备位置信息到配置文件 - 不再需要，因为位置信息不可编辑"""
        # 此功能已被禁用，只保留函数结构以保持代码兼容性
        pass

    def is_location_modified(self):
        """检查设备位置信息是否被修改 - 不再需要，位置信息不可编辑"""
        # 位置信息不可编辑，始终返回False
        return False

    def handle_exit(self):
        """统一处理退出逻辑"""
        # 位置信息不可编辑，无需检查修改
        self.close()  # 关闭窗口

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        self.handle_exit()  # 调用统一退出逻辑
        event.accept()  # 允许窗口关闭

    def init_main_tab2(self):
        """初始化主界面2布局，复制主界面的全部功能"""
        layout = QVBoxLayout()
        font = QFont("Arial", 16)
        self.exit_button2 = QPushButton("退出")
        self.exit_button2.clicked.connect(self.handle_exit)  # 绑定退出逻辑
        self.label2 = QLabel("")  # 删除"设备状态: "文本
        self.label2.setFont(font)
        layout.addWidget(self.label2)

        # 添加设备状态表格
        self.status_table2 = QTableWidget()
        self.status_table2.setColumnCount(8)  # 从9列减少到8列，移除"位置编号"列
        self.status_table2.setHorizontalHeaderLabels(["绝缘子串ID", "线路电压", "设备状态", "最新数据时间", "温度", "湿度", "绝缘程度", "绝缘状态"])
        self.status_table2.setFont(font)
        layout.addWidget(self.status_table2)

        # 初始化控制按钮
        self.control_layout2 = QHBoxLayout()

        self.predict_button2 = QPushButton("刷新数据")
        self.predict_button2.setFont(font)
        self.predict_button2.clicked.connect(self.update_predictions)  # 使用相同的更新功能
        self.control_layout2.addWidget(self.predict_button2)

        self.exit_button2 = QPushButton("退出")
        self.exit_button2.setFont(font)
        self.exit_button2.clicked.connect(self.close)
        self.control_layout2.addWidget(self.exit_button2)

        layout.addLayout(self.control_layout2)

        # 设置主界面2布局
        self.main_tab2.setLayout(layout)

        # 初始化设备状态表格内容
        self.initialize_table2()
        
    def initialize_table2(self):
        """初始化设备状态表格2"""
        self.status_table2.setRowCount(len(self.devices))
        
        # 设置字体
        font = QFont("Arial", 14)  # 字体 Arial，大小 14
        self.status_table2.setFont(font)  # 统一设置表格字体

        # 初始化设备表格内容
        for i, device_name in enumerate(self.devices):
            # 设置设备名称列
            item = QTableWidgetItem(str(device_name))
            item.setData(Qt.UserRole, i)  # 存储设备索引
            item.setFlags(item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.status_table2.setItem(i, 0, item)

            # 初始化其他列
            for col in range(1, 9):  # 9列
                item = QTableWidgetItem("-")
                self.status_table2.setItem(i, col, item)

        # 设置列宽和标题
        self.status_table2.setColumnCount(8)  # 8列
        self.status_table2.setHorizontalHeaderLabels(["绝缘子串ID", "线路电压", "设备状态", "最新数据时间", "温度", "湿度", "绝缘程度", "绝缘状态"])
        self.status_table2.setColumnWidth(0, 350)  # 绝缘子串ID列宽度
        self.status_table2.setColumnWidth(1, 120)  # 线路电压列宽度
        self.status_table2.setColumnWidth(2, 120)  # 设备状态列宽度
        self.status_table2.setColumnWidth(3, 300)  # 时间列宽度
        self.status_table2.setColumnWidth(4, 120)  # 温度列宽度
        self.status_table2.setColumnWidth(5, 120)  # 湿度列宽度
        self.status_table2.setColumnWidth(6, 180)  # 绝缘程度列宽度
        self.status_table2.setColumnWidth(7, 180)  # 绝缘状态列宽度

        # 添加点击事件跳转到设备详情
        self.status_table2.itemClicked.connect(self.on_device_name_clicked)

    def eventFilter(self, obj, event):
        """全局事件过滤器，用于捕获快捷键"""
        if event.type() == QEvent.KeyPress:
            # 检测Ctrl+Z组合键
            if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
                self.toggle_admin_mode()
                return True  # 事件已处理
        return super().eventFilter(obj, event)  # 其他事件交给父类处理

    def toggle_admin_mode(self):
        """切换管理员模式，需要密码验证"""
        if self.admin_mode:
            # 如果当前是管理员模式，直接切换到普通模式
            self.admin_mode = False
            self.update_admin_mode_ui()
            QMessageBox.information(self, "模式切换", "已退出管理员模式")
        else:
            # 如果当前是普通模式，需要输入密码验证
            password, ok = QInputDialog.getText(
                self, "管理员验证", "请输入管理员密码:", 
                QLineEdit.Password
            )
            if ok and password == self.admin_password:
                self.admin_mode = True
                self.update_admin_mode_ui()
                QMessageBox.information(self, "模式切换", "已进入管理员模式")
            elif ok:  # 密码错误但用户点击了确定
                QMessageBox.warning(self, "验证失败", "密码错误，无法进入管理员模式")

    def update_admin_mode_ui(self):
        """根据当前管理员模式状态更新界面"""
        if self.admin_mode:
            # 管理员模式下，添加主界面2标签页
            if self.tabs.indexOf(self.main_tab2) == -1:  # 检查是否已添加
                self.tabs.insertTab(1, self.main_tab2, "主界面2")  # 在设备详情前添加
        else:
            # 普通模式下，移除主界面2标签页
            index = self.tabs.indexOf(self.main_tab2)
            if index != -1:  # 确保存在再移除
                self.tabs.removeTab(index)
                # 如果当前正在显示主界面2，则切换到主界面
                if self.tabs.currentWidget() == self.main_tab2:
                    self.tabs.setCurrentWidget(self.main_tab)


# 运行应用
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
