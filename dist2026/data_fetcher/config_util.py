import configparser
import json

class Config:
    def __init__(self, filename):
        self.filename = filename
        self.config = configparser.ConfigParser()
        self.config.read(self.filename, encoding="UTF-8")

    def get_config(self, section, index):
        """获取配置项的值"""
        return self.config.get(section, index, fallback='NOT FOUND')

    def get_config_bool(self, section, index):
        """获取布尔值配置项"""
        return self.config.getboolean(section, index, fallback=False)

    def get_config_int(self, section, index):
        """获取整型值配置项"""
        return self.config.getint(section, index, fallback=0)

    def get_config_list(self, section, index):
        """获取列表值配置项（以JSON格式存储）"""
        value = self.config.get(section, index, fallback='[]')
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            # 尝试修复常见的 JSON 格式错误
            # 移除尾随逗号
            value_cleaned = value.strip()
            # 处理尾随逗号的情况，如 [1, 2,]
            import re
            # 移除数组中的尾随逗号
            value_cleaned = re.sub(r',\s*]', ']', value_cleaned)
            value_cleaned = re.sub(r',\s*}', '}', value_cleaned)
            try:
                return json.loads(value_cleaned)
            except json.JSONDecodeError:
                # 如果仍然失败，抛出更详细的错误信息
                raise ValueError(f"配置文件 {self.filename} 中 [{section}] 节的 {index} 值不是有效的 JSON 格式: {value}. 错误: {e}")

    def get_sections(self):
        """获取所有配置节"""
        sections = ["DEFAULT"]
        sections.extend(self.config.sections())
        return sections

    def has_option(self, section, index):
        """检查某节中是否存在某项"""
        return self.config.has_option(section, index)

    def set_config(self, section, index, value):
        """设置配置项"""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, index, value)

    def write_config(self):
        """将配置写入文件"""
        with open(self.filename, "w", encoding="UTF-8") as configfile:
            self.config.write(configfile)
