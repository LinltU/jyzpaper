import time
import requests
import pandas as pd
import os
import json
from .bitable import Bitable
from .config_util import Config

class DataFetcher:
    def __init__(self, config_file=None, output_folder=None):
        """
        初始化 DataFetcher 类，从配置文件中加载必要的参数。
        
        :param config_file: 配置文件路径，默认为 "config.ini"
        """
        if config_file is None:
            config_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "config.ini",
            )
        self.config = Config(config_file)
        config_dir = os.path.dirname(os.path.abspath(config_file))
        self.output_folder = output_folder or os.path.join(os.path.dirname(config_dir), "output")

        # # 从配置文件中加载各项参数
        # self.BASE_URL = self.config.get_config("DEFAULT", "BASE_URL")
        # self.username = self.config.get_config("DEFAULT", "username")
        # self.password = self.config.get_config("DEFAULT", "password")
        # self.LOGIN_DATA = {"username": self.username, "password": self.password}
        # self.collect_all_devices = self.config.get_config_bool("QUERY", "collect_all_devices")
        # self.table_id = self.config.get_config("DEFAULT", "table_id")
        # self.app_id = self.config.get_config("DEFAULT", "app_id")
        # self.app_secret = self.config.get_config("DEFAULT", "app_secret")
        # self.wiki_id = self.config.get_config("DEFAULT", "wiki_id")
        
        # # 从配置中获取查询时间范围和设备列表
        # self.start_time = self.config.get_config("QUERY", "start_time")
        # self.end_time = self.config.get_config("QUERY", "end_time")
        # self.devices = self.config.get_config_list("QUERY", "devices")
        # self.table_name = self.config.get_config("QUERY", "table_name")


        # 连接信息和凭据仅从本地配置读取，不在源码中保存默认凭据。
        self.BASE_URL = self.config.get_config("DEFAULT", "BASE_URL")
        self.username = self.config.get_config("DEFAULT", "username")
        self.password = self.config.get_config("DEFAULT", "password")
        for name in ("BASE_URL", "username", "password"):
            if not getattr(self, name) or getattr(self, name) == "NOT FOUND":
                raise ValueError(f"Missing DEFAULT.{name} in {config_file}")

        self.LOGIN_DATA = {"username": self.username, "password": self.password}

        # 飞书参数同样从被 Git 忽略的本地配置读取。
        self.collect_all_devices = False
        self.table_id = self.config.get_config("DEFAULT", "table_id")
        self.app_id = self.config.get_config("DEFAULT", "app_id")
        self.app_secret = self.config.get_config("DEFAULT", "app_secret")
        self.wiki_id = self.config.get_config("DEFAULT", "wiki_id")
        
        # 直接获取查询时间范围和设备列表
        self.start_time = self.config.get_config("QUERY", "start_time")
        self.end_time = self.config.get_config("QUERY", "end_time")
        self.devices = self.config.get_config_list("QUERY", "devices")
        self.table_name = self.config.get_config("QUERY", "table_name")
    def time_convert(self, t):
        """
        将时间戳和格式化时间字符串相互转换（与 dist 中保持一致）。
        
        :param t: 可以是时间戳（int）或格式化时间字符串（str）
        :return: 转换后的时间戳或格式化时间字符串
        """
        if isinstance(t, int):
            tmp = t / 1000
            time_local = time.localtime(tmp)
            ans = time.strftime("%Y-%m-%d_%H:%M:%S", time_local)
        else:
            tmp = time.strptime(t, "%Y-%m-%d_%H:%M:%S")
            ans = int(time.mktime(tmp)) * 1000
        return ans

    def login(self):
        """ 登录并返回 session 对象（按 dist 目录中的逻辑，检查是否跳转到 /index）"""
        try:
            session = requests.Session()
            response = session.post(f"{self.BASE_URL}/login", data=self.LOGIN_DATA)
            if response.url.endswith('/index'):
                print("Login successful!")
                return session
            else:
                print("Login failed.")
                session.close()  # 关闭失败的会话
                return None
        except Exception as e:
            print(f"Login error: {str(e)}")
            return None

    def get_imei_numbers(self, session):
        """ 获取设备的 IMEI 列表（与 dist 中保持一致） """
        response = session.get(f"{self.BASE_URL}/get_imei_numbers")
        if response.status_code == 200:
            return response.json()['imeiNumbers']
        else:
            print("Failed to get IMEI numbers")
            return []

    def get_runtimes(self, session, imei):
        """ 获取设备的运行时数据（按照 dist 目录中的接口实现） """
        payload = {"selected_imei": imei}
        response = session.post(f"{self.BASE_URL}/get_runtimes", data=payload)
        if response.status_code == 200:
            # 假设返回的数据已按时间戳排序（从新到旧），若未排序则在此进行排序
            runtimes = response.json()
            # 确保按时间戳从新到旧排序
            runtimes.sort(reverse=True)
            return runtimes
        else:
            print(f"Failed to get runtimes for IMEI {imei}")
            return []

    def get_timed_status(self, session, imei, runtime):
        """ 获取设备在特定运行时的状态（使用 /get_timed_status 接口，与 dist 中保持一致） """
        response = session.get(f"{self.BASE_URL}/get_timed_status/{imei}/{runtime}")
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Failed to get timed status for IMEI {imei}"
            print(error_msg)
            raise Exception(error_msg)  # 按 dist 中逻辑抛出异常而不是返回 None

    def save_to_excel_per_device(self, all_data, folder=None, filename="output.xlsx"):
        """
        将每个设备的数据保存到 Excel 文件中，设备数据存储在不同的 sheet 中。
        
        :param all_data: 所有设备的数据，key 为 IMEI，value 为设备数据列表。
        :param folder: 存储文件的文件夹，默认为 "output"。
        :param filename: Excel 文件的名称，默认为 "output.xlsx"。
        """
        folder = folder or self.output_folder

        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 获取当前时间戳，用于文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        final_filename = os.path.join(folder, f"{filename}_{timestamp}.xlsx")

        with pd.ExcelWriter(final_filename, engine='openpyxl') as writer:
            for imei, device_data in all_data.items():
                df = pd.DataFrame(device_data, columns=[
                    "IMEI", "时间", "电池电压", "测量值", "温度", "湿度", "版本号"
                ])

                # 将列标签翻译为英文
                df.columns = [
                    "IMEI", "Timestamp", "Battery", "Measure", "Temp", "Hum", "Version"
                ]

                # 确保 "电池电压", "测量值", "温度", "湿度" 列数据格式为 float
                df["Battery"] = df["Battery"].astype(float)
                df["Measure"] = df["Measure"].astype(float)
                df["Temp"] = df["Temp"].astype(float)
                df["Hum"] = df["Hum"].astype(float)

                # 确保时间列格式为指定的 "yyyy-mm-dd_HH:MM:SS"
                df["Timestamp"] = df["Timestamp"].apply(lambda x: self.time_convert(x))

                # 将 DataFrame 写入 Excel 中
                df.to_excel(writer, index=False, sheet_name=str(imei))
        print(f"数据已成功保存到 {final_filename}")

    def fetch_data(self):
        """ 从服务器获取数据并保存到本地 Excel 文件 """
        session = self.login()
        if not session:
            raise Exception("登录失败，无法获取数据")

        try:
            imei_numbers = self.get_imei_numbers(session)

            # 转换时间格式
            start_time = self.time_convert(self.start_time)
            end_time = self.time_convert(self.end_time)

            all_data = {}  # 用来存储每个设备的数据，以 IMEI 为键

            # 根据配置获取设备列表
            devices_to_process = imei_numbers if self.collect_all_devices else self.devices

            # 遍历每个设备，获取其状态信息
            for imei in devices_to_process:
                runtimes = self.get_runtimes(session, imei)
                print(f"正在上传设备 #{imei}...")

                device_data = []  # 存储当前设备的数据

                for runtime in runtimes:
                    unix = self.time_convert(runtime)
                    # 检查时间范围：数据时间戳必须在 start_time 和 end_time 之间
                    if start_time <= unix <= end_time:
                        status = self.get_timed_status(session, imei, runtime)
                        if status:
                            analog = float(status["Analog"])
                            battery = float(status["Battrey"])
                            humidity = float(status["Humidity"])
                            version = status["LocalVersion"]
                            temperature = float(status["Temperature"])
                            analog = "{:.4f}".format(analog)
                            battery = "{:.3f}".format(battery)
                            humidity = "{:.3f}".format(humidity)
                            temperature = "{:.3f}".format(temperature)
                            content = [str(imei), unix, str(battery), str(analog), str(temperature), str(humidity), str(version)]
                            device_data.append(content)

                if device_data:
                    all_data[imei] = device_data  # 保存当前设备的数据

            # 将数据保存到 Excel 文件
            self.save_to_excel_per_device(all_data)
            print("数据上传并保存到 Excel 完成！")
        finally:
            if session:
                session.close()

    def get_latest_data(self, imei):
        """
        获取指定设备最新的数据（即最新的一条记录），并返回 Measure、Temp、Hum 和 时间戳。
        
        :param imei: 设备的 IMEI 编号
        :return: 最新的数据：Measure、Temp、Hum 和 时间戳
        """
        session = self.login()
        if not session:
            raise Exception(f"登录失败，无法获取设备 {imei} 数据")

        try:
            # 获取设备的运行时数据
            runtimes = self.get_runtimes(session, imei)
            if not runtimes:
                error_msg = f"设备 {imei} 没有找到任何运行时数据"
                print(error_msg)
                raise Exception(error_msg)

            # 获取最新的一条数据
            latest_runtime = runtimes[0]  # 假设最新的数据在列表的开头
            status = self.get_timed_status(session, imei, latest_runtime)

            if status:
                try:
                    # 打印状态数据以便调试
                    print(f"解析状态数据，类型: {type(status)}")
                    if isinstance(status, dict):
                        print(f"状态数据字段: {list(status.keys())}")
                    else:
                        print(f"状态数据: {status}")
                    
                    # 检查必要的字段是否存在
                    if not isinstance(status, dict):
                        raise Exception(f"状态数据不是字典类型: {type(status)}, 数据: {status}")
                    
                    missing_fields = []
                    if "Analog" not in status:
                        missing_fields.append("Analog")
                    if "Temperature" not in status:
                        missing_fields.append("Temperature")
                    if "Humidity" not in status:
                        missing_fields.append("Humidity")
                    if "Battrey" not in status:
                        missing_fields.append("Battrey")
                    if "LocalVersion" not in status:
                        missing_fields.append("LocalVersion")
                    
                    if missing_fields:
                        error_msg = f"响应数据中缺少必要的字段: {missing_fields}"
                        print(error_msg)
                        print(f"可用字段: {list(status.keys())}")
                        raise Exception(error_msg)
                    
                    analog = float(status["Analog"])
                    temperature = float(status["Temperature"])
                    humidity = float(status["Humidity"])
                    battery = float(status["Battrey"])
                    version = status["LocalVersion"]

                    print(f"成功解析数据: Analog={analog}, Temperature={temperature}, Humidity={humidity}, Battery={battery}")
                    
                    # 返回最新的 Measure, Temp, Hum 和 时间戳 (latest_runtime)
                    return analog, temperature, humidity, latest_runtime, battery, version 
                except ValueError as e:
                    error_msg = f"数据转换为 float 时发生错误: {e}"
                    print(error_msg)
                    if isinstance(status, dict):
                        print(f"原始数据: Analog={status.get('Analog')}, Temperature={status.get('Temperature')}, Humidity={status.get('Humidity')}, Battery={status.get('Battrey')}")
                    raise Exception(error_msg)
                except KeyError as e:
                    error_msg = f"响应数据中缺少必要的字段: {e}"
                    print(error_msg)
                    print(f"可用字段: {list(status.keys()) if isinstance(status, dict) else '非字典类型'}")
                    raise Exception(error_msg)
            else:
                error_msg = f"设备 {imei} 没有获取到最新的状态数据"
                print(error_msg)
                raise Exception(error_msg)
        finally:
            if session:
                session.close()

    def get_all_latest_data(self):
        """
        获取所有配置设备的最新数据，并按 config.ini 中的设备顺序返回。
        
        :return: 按设备顺序排列的最新数据数组，包含每个设备的 Measure、Temp、Hum 和 时间戳。
        """
        latest_data_list = []  # 存储所有设备的最新数据

        # 遍历所有设备，根据配置文件中的设备顺序获取最新数据
        for imei in self.devices:
            latest_data = self.get_latest_data(imei)
            if latest_data:
                latest_data_list.append(latest_data)
            else:
                latest_data_list.append((None, None, None, None, None, None))  # 如果某个设备没有数据，返回 None

        return latest_data_list

    def get_last_n_data(self, device_name, n=15):
        """获取指定设备最近n条数据"""
        session = self.login()
        if not session:
            raise Exception(f"登录失败，无法获取设备 {device_name} 数据")

        try:
            runtimes = self.get_runtimes(session, device_name)
            data_list = []
            
            # 获取前n个runtime的数据（runtimes已按时间从新到旧排序）
            for runtime in runtimes[:n]:
                try:
                    status = self.get_timed_status(session, device_name, runtime)
                    if status:
                        try:
                            analog = float(status["Analog"])
                            temp = float(status["Temperature"])
                            hum = float(status["Humidity"])
                            battery = float(status["Battrey"])
                            version = status["LocalVersion"]
                            data_list.append((analog, temp, hum, runtime, battery, version))
                        except (KeyError, ValueError) as e:
                            print(f"数据解析错误: {e}")
                            continue
                except Exception as e:
                    # 将异常传递给调用者，不在这里处理
                    print(f"获取设备 {device_name} 状态时出错: {str(e)}")
                    raise
            
            # 返回最近n条数据（按收集顺序，最新的数据在前面）
            return data_list
        finally:
            if session:
                session.close()

    def get_all_last_n_data(self, n=10):
        """获取所有设备最近n条数据"""
        all_data = []
        for device in self.devices:
            try:
                device_data = self.get_last_n_data(device, n)
                all_data.append(device_data)
            except Exception as e:
                # 将异常传递给调用者
                print(f"获取设备 {device} 的最近 {n} 条数据失败: {e}")
                raise
        return all_data
