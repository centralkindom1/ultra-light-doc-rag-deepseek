import sys
import os
import json
import re
import requests
import time
import threading
import urllib3
import traceback
from datetime import datetime
from typing import List, Dict

# 禁用 urllib3 的安全警告 (针对 Win7/局域网自签名证书)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. 依赖库检查与导入
# ==========================================
try:
    from docx import Document
    from sentence_transformers import SentenceTransformer
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                                 QFileDialog, QListWidget, QSplitter, QMessageBox, 
                                 QProgressBar, QGroupBox, QTableWidget, QTableWidgetItem, 
                                 QHeaderView, QAbstractItemView)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
    from PyQt5.QtGui import QFont, QColor, QTextCursor, QTextCharFormat
except ImportError as e:
    print("【严重错误】缺少必要的库，请先运行 pip install python-docx PyQt5 sentence-transformers requests")
    print(f"详细错误: {e}")
    sys.exit(1)

# ==========================================
# 1. 核心配置区 (User Config)
# ==========================================
CONFIG = {
    "API_URL": "https://aiplus.airchina.com.cn:18080/v1/chat/completions",
    "API_KEY": "sk-fXM4W0CdcKnNp3NVDfF85f2b90284b11AfDdF9F5627f627b",
    "API_MODEL": "qwen2.5-72b",  # 局域网模型名称
    "BGE_PATH": r"D:\Models\bge-small-zh-v1.5", # 本地 BGE 模型路径
    "MAX_SAMPLE_CHARS": 2500,    # 发送给 AI 进行分析的字符数
    "APP_TITLE": "DRS - Docx Rule Studio (Industrial Win7 Edition)",
    "FONT_FAMILY": "Microsoft YaHei" # Win7 友好字体
}

# ==========================================
# 2. 基础设施层 (Logging & Signals)
# ==========================================
class LogSignal(QObject):
    """用于跨线程发送日志信号"""
    text_written = pyqtSignal(str, str) # content, color

class OutputStream(object):
    """重定向 stdout/stderr 到 GUI"""
    def __init__(self, signal_emitter, color="white"):
        self.emitter = signal_emitter
        self.color = color

    def write(self, text):
        if text.strip():
            self.emitter.text_written.emit(str(text), self.color)

    def flush(self):
        pass

# ==========================================
# 3. 后台工作线程 (Workers)
# ==========================================

class ModelLoaderWorker(QThread):
    """异步加载 BGE 模型，防止启动时界面卡死"""
    finished_signal = pyqtSignal(object)
    log_signal = pyqtSignal(str)

    def run(self):
        self.log_signal.emit(f"正在加载本地 Embedding 模型: {CONFIG['BGE_PATH']} ...")
        t_start = time.time()
        try:
            if os.path.exists(CONFIG['BGE_PATH']):
                # 强制 CPU 模式
                model = SentenceTransformer(CONFIG['BGE_PATH'], device='cpu')
                self.log_signal.emit(f"模型加载成功! 耗时: {time.time() - t_start:.2f}秒")
                self.finished_signal.emit(model)
            else:
                self.log_signal.emit(f"⚠️ 警告: 路径不存在 {CONFIG['BGE_PATH']}，跳过加载。")
                self.finished_signal.emit(None)
        except Exception as e:
            self.log_signal.emit(f"❌ 模型加载失败: {str(e)}")
            self.finished_signal.emit(None)

class AIAnalysisWorker(QThread):
    """异步调用局域网 LLM API"""
    result_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    
    def __init__(self, text_sample):
        super().__init__()
        self.text_sample = text_sample

    def run(self):
        self.log_signal.emit(">>> 开始构建 AI 请求 Prompt...")
        
        prompt = f"""
你是一个 RAG 数据清洗专家。请分析以下从 DOCX 文档提取的文本片段（包含元数据）。
你的任务是识别出“噪音”内容（如：页眉、页脚、导航菜单、日期戳、广告、无意义的分隔符）。

【待分析文本片段】：
{self.text_sample}

【要求】：
1. 仔细观察文本的模式。例如 "| 下一章节 |" 这种明显是导航。
2. 给出能匹配这些噪音的 Python 正则表达式 (Regex) 列表。
3. 给出识别正文标题的关键词建议。

请严格仅返回以下 JSON 格式，不要包含 Markdown 代码块标记（```json）：
{{
  "noise_regex": [
    "正则表达式1",
    "正则表达式2"
  ],
  "heading_hints": ["关键词1", "关键词2"],
  "analysis_summary": "简短的分析说明"
}}
"""
        headers = {
            'Content-Type': 'application/json', 
            'Authorization': f"Bearer {CONFIG['API_KEY']}"
        }
        payload = {
            "model": CONFIG['API_MODEL'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, # 低温度保证正则的准确性
            "stream": False
        }

        try:
            self.log_signal.emit(f"POST {CONFIG['API_URL']} (verify=False)")
            t_start = time.time()
            response = requests.post(
                CONFIG['API_URL'], 
                headers=headers, 
                json=payload, 
                verify=False, # 忽略 SSL
                timeout=60
            )
            elapsed = time.time() - t_start
            self.log_signal.emit(f"AI 响应接收完成，耗时: {elapsed:.2f}s, 状态码: {response.status_code}")

            if response.status_code == 200:
                raw_content = response.json()['choices'][0]['message']['content']
                self.log_signal.emit("AI 原始返回: " + raw_content[:100] + "...")
                
                # 鲁棒性 JSON 提取
                json_str = raw_content
                # 尝试去除 markdown 标记
                match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if match:
                    json_str = match.group()
                
                try:
                    data = json.loads(json_str)
                    self.result_signal.emit(data)
                except json.JSONDecodeError:
                    self.log_signal.emit("❌ JSON 解析失败，AI 返回格式不正确。")
                    self.result_signal.emit({})
            else:
                self.log_signal.emit(f"❌ API 错误: {response.text}")
                self.result_signal.emit({})

        except Exception as e:
            self.log_signal.emit(f"❌ 网络请求异常: {str(e)}")
            self.log_signal.emit(traceback.format_exc())
            self.result_signal.emit({})

# ==========================================
# 4. 业务逻辑层 (Docx Engine)
# ==========================================
class DocxEngine:
    def __init__(self):
        self.embedding_model = None

    def parse_file(self, file_path):
        """解析 DOCX，返回结构化数据列表"""
        doc = Document(file_path)
        extracted_data = []
        
        # 遍历段落
        for i, p in enumerate(doc.paragraphs):
            text = p.text.strip()
            if not text:
                continue
            
            # 提取元数据特征
            style_name = p.style.name
            is_bold = any(run.bold for run in p.runs)
            
            # 简单判断是否可能是导航/页眉 (启发式)
            # 例如：包含 '|' 且长度短，或者全是日期格式
            heuristic_tag = "Text"
            if "|" in text and len(text) < 50:
                heuristic_tag = "Nav/Noise?"
            
            extracted_data.append({
                "index": i,
                "text": text,
                "style": style_name,
                "bold": is_bold,
                "tag": heuristic_tag
            })
            
        return extracted_data

    def apply_cleaning(self, data_list: List[Dict], regex_rules: List[str]) -> List[str]:
        """应用正则规则清洗文本"""
        cleaned_lines = []
        total_removed = 0
        
        compiled_rules = []
        for r in regex_rules:
            try:
                compiled_rules.append(re.compile(r))
            except re.error as e:
                print(f"无效的正则: {r} ({e})")

        for item in data_list:
            original_text = item['text']
            temp_text = original_text
            
            # 1. 完全匹配模式 (针对整行噪音)
            is_noise = False
            for pattern in compiled_rules:
                # 如果正则匹配了大部分文本，或者是特定的替换
                if pattern.search(temp_text):
                    # 这里做简单的“替换为空”策略
                    temp_text = pattern.sub("", temp_text).strip()
            
            if temp_text:
                cleaned_lines.append(temp_text)
            else:
                total_removed += 1
                
        return cleaned_lines, total_removed

# ==========================================
# 5. GUI 表现层 (Main Window)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = DocxEngine()
        self.raw_data = [] # 存储解析后的结构化数据
        self.init_ui()
        self.setup_logging()
        
        # 启动时自动加载模型
        self.load_model_thread()

    def setup_logging(self):
        """配置控制台重定向"""
        self.log_signal = LogSignal()
        self.log_signal.text_written.connect(self.append_log)
        sys.stdout = OutputStream(self.log_signal, color="#00FF00") # 绿色标准输出
        sys.stderr = OutputStream(self.log_signal, color="#FF5555") # 红色错误输出

    def append_log(self, text, color_code):
        """向底部控制台追加文本"""
        cursor = self.console_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color_code))
        
        # 添加时间戳
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        cursor.insertText(timestamp, fmt)
        cursor.insertText(text + "\n", fmt)
        
        self.console_output.setTextCursor(cursor)
        self.console_output.ensureCursorVisible()

    def init_ui(self):
        self.setWindowTitle(CONFIG['APP_TITLE'])
        self.resize(1280, 850)
        self.setStyleSheet(f"font-family: '{CONFIG['FONT_FAMILY']}'; font-size: 10pt;")

        # 主 Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 顶部工具栏 ---
        top_bar = QGroupBox("流水线控制")
        top_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("📂 1. 加载 DOCX 文档")
        self.btn_load.clicked.connect(self.action_load_file)
        
        self.btn_ai = QPushButton("🤖 2. 调用局域网 AI 生成规则")
        self.btn_ai.clicked.connect(self.action_ai_analyze)
        self.btn_ai.setEnabled(False) # 初始禁用
        
        self.btn_preview = QPushButton("▶️ 3. 执行清洗预览")
        self.btn_preview.clicked.connect(self.action_preview_clean)
        
        self.btn_export = QPushButton("💾 4. 导出规则 JSON")
        self.btn_export.clicked.connect(self.action_export)

        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.btn_ai)
        top_layout.addWidget(self.btn_preview)
        top_layout.addWidget(self.btn_export)
        top_bar.setLayout(top_layout)
        main_layout.addWidget(top_bar)

        # --- 中间核心区 (Splitter) ---
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：原始结构化视图 (Table)
        left_group = QGroupBox("原始文档特征 (Source)")
        left_layout = QVBoxLayout()
        self.table_raw = QTableWidget()
        self.table_raw.setColumnCount(3)
        self.table_raw.setHorizontalHeaderLabels(["Style", "Text Content", "Tag"])
        self.table_raw.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_raw.setSelectionBehavior(QAbstractItemView.SelectRows)
        left_layout.addWidget(self.table_raw)
        left_group.setLayout(left_layout)

        # 中间：规则编辑器 (List)
        mid_group = QGroupBox("清洗规则集 (Rule Studio)")
        mid_layout = QVBoxLayout()
        self.list_rules = QListWidget()
        self.list_rules.setAlternatingRowColors(True)
        
        btn_add_rule = QPushButton("➕ 手动添加正则")
        btn_add_rule.clicked.connect(self.action_add_rule)
        btn_del_rule = QPushButton("➖ 删除选中规则")
        btn_del_rule.clicked.connect(self.action_del_rule)
        
        mid_layout.addWidget(QLabel("AI 建议或手动输入的正则:"))
        mid_layout.addWidget(self.list_rules)
        mid_layout.addWidget(btn_add_rule)
        mid_layout.addWidget(btn_del_rule)
        mid_group.setLayout(mid_layout)

        # 右侧：清洗结果 (Text)
        right_group = QGroupBox("纯净文本 (Cleaned Target)")
        right_layout = QVBoxLayout()
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setStyleSheet("background-color: #f8fcf8; color: #333;")
        right_layout.addWidget(self.text_preview)
        right_group.setLayout(right_layout)

        splitter.addWidget(left_group)
        splitter.addWidget(mid_group)
        splitter.addWidget(right_group)
        splitter.setSizes([450, 300, 450])
        main_layout.addWidget(splitter, stretch=2)

        # --- 底部控制台 ---
        console_group = QGroupBox("系统交互日志 (System Console)")
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            background-color: #1e1e1e; 
            color: #00FF00; 
            font-family: 'Consolas', 'Courier New'; 
            font-size: 9pt;
        """)
        console_layout.addWidget(self.console_output)
        console_group.setLayout(console_layout)
        main_layout.addWidget(console_group, stretch=1)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪 - 等待操作")

    # ==========================================
    # 逻辑动作函数
    # ==========================================
    
    def load_model_thread(self):
        """后台加载模型"""
        self.loader_thread = ModelLoaderWorker()
        self.loader_thread.log_signal.connect(lambda msg: print(msg)) # 打印到控制台
        self.loader_thread.finished_signal.connect(self.on_model_loaded)
        self.loader_thread.start()

    def on_model_loaded(self, model):
        self.engine.embedding_model = model
        if model:
            self.status_bar.showMessage(f"Embedding 模型已就绪: {CONFIG['BGE_PATH']}")
        else:
            self.status_bar.showMessage("模型加载失败，将无法计算向量，但不影响正则清洗。")

    def action_load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 Docx 文件", "", "Word Documents (*.docx)")
        if not path:
            return

        print(f"正在读取文档: {path} ...")
        try:
            self.raw_data = self.engine.parse_file(path)
            self.update_raw_table()
            self.btn_ai.setEnabled(True)
            print(f"读取完成，共提取 {len(self.raw_data)} 个文本段落。")
            self.status_bar.showMessage(f"已加载: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ 读取错误: {e}")

    def update_raw_table(self):
        self.table_raw.setRowCount(0)
        self.table_raw.setRowCount(len(self.raw_data))
        for row, item in enumerate(self.raw_data):
            # Style Cell
            item_style = QTableWidgetItem(item['style'])
            if "Heading" in item['style']:
                item_style.setForeground(QColor("blue"))
            self.table_raw.setItem(row, 0, item_style)
            
            # Text Cell
            text_preview = item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
            item_text = QTableWidgetItem(text_preview)
            if item['bold']:
                font = QFont()
                font.setBold(True)
                item_text.setFont(font)
            self.table_raw.setItem(row, 1, item_text)
            
            # Tag Cell
            self.table_raw.setItem(row, 2, QTableWidgetItem(item['tag']))

    def action_ai_analyze(self):
        if not self.raw_data:
            return
        
        # 构造样本：取前 N 个字符，并且带上样式标记，方便 AI 判断
        sample_lines = []
        char_count = 0
        for item in self.raw_data:
            line = f"[{item['style']}] {item['text']}"
            sample_lines.append(line)
            char_count += len(line)
            if char_count > CONFIG['MAX_SAMPLE_CHARS']:
                break
        
        sample_text = "\n".join(sample_lines)
        
        self.btn_ai.setEnabled(False)
        self.status_bar.showMessage("AI 正在思考中...")
        
        # 启动 AI 线程
        self.ai_thread = AIAnalysisWorker(sample_text)
        self.ai_thread.log_signal.connect(lambda msg: print(msg))
        self.ai_thread.result_signal.connect(self.on_ai_result)
        self.ai_thread.start()

    def on_ai_result(self, result_dict):
        self.btn_ai.setEnabled(True)
        self.status_bar.showMessage("AI 分析完成")
        
        if not result_dict:
            QMessageBox.warning(self, "AI 失败", "未能获取有效的 JSON 规则，请检查控制台日志。")
            return

        regex_list = result_dict.get("noise_regex", [])
        hints = result_dict.get("heading_hints", [])
        summary = result_dict.get("analysis_summary", "无说明")

        print(f"------ AI 建议 ------\n摘要: {summary}\n正则数量: {len(regex_list)}")
        
        # 填充到界面
        self.list_rules.clear()
        for r in regex_list:
            self.list_rules.addItem(r)
            
        QMessageBox.information(self, "AI 分析成功", f"AI 已生成 {len(regex_list)} 条清洗规则。\n\n分析摘要：{summary}")

    def action_add_rule(self):
        self.list_rules.addItem("在此输入新正则...")
        item = self.list_rules.item(self.list_rules.count() - 1)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.list_rules.editItem(item)

    def action_del_rule(self):
        row = self.list_rules.currentRow()
        if row >= 0:
            self.list_rules.takeItem(row)

    def action_preview_clean(self):
        if not self.raw_data:
            return
        
        rules = [self.list_rules.item(i).text() for i in range(self.list_rules.count())]
        print(f"正在应用 {len(rules)} 条正则规则...")
        
        t_start = time.time()
        cleaned_text_list, removed_count = self.engine.apply_cleaning(self.raw_data, rules)
        
        # 显示结果
        full_text = "\n\n".join(cleaned_text_list)
        self.text_preview.setText(full_text)
        
        elapsed = time.time() - t_start
        print(f"清洗完成。删除了 {removed_count} 行噪音，耗时 {elapsed:.4f}s")
        self.status_bar.showMessage(f"预览已更新 (保留 {len(cleaned_text_list)} 行)")

    def action_export(self):
        rules = [self.list_rules.item(i).text() for i in range(self.list_rules.count())]
        if not rules:
            QMessageBox.warning(self, "提示", "规则列表为空，无法导出。")
            return
            
        save_path = "industrial_rules.json"
        export_data = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "bge_model": CONFIG['BGE_PATH'],
            "regex_rules": rules
        }
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"规则已保存至: {os.path.abspath(save_path)}")
            QMessageBox.information(self, "导出成功", f"工业级规则已保存至:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

# ==========================================
# 6. 程序入口 (Main)
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 针对高分屏的优化 (可选)
    # app.setAttribute(Qt.AA_EnableHighDpiScaling) 
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())