import sys
import os
import json
import sqlite3
import time
import requests
import urllib3
import traceback
import numpy as np  # 核心数学库：用于向量矩阵运算
from typing import List, Dict, Tuple

# 禁用 urllib3 安全警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. 依赖与配置
# ==========================================
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                                 QFileDialog, QSplitter, QMessageBox, QProgressBar, 
                                 QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, 
                                 QAbstractItemView, QComboBox, QDoubleSpinBox, QSpinBox,
                                 QStyleFactory, QFrame)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
    from PyQt5.QtGui import QFont, QColor, QBrush
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print("【严重错误】缺少库，请运行: pip install PyQt5 sentence-transformers numpy requests")
    sys.exit(1)

# --- 全局配置 (请保持与 v4 生成端一致) ---
GLOBAL_CONFIG = {
    # 远程 Embedding API
    "EMBED_API_URL": "https://www.deepseek.com:18080/v1/embeddings",
    "REMOTE_MODEL_NAME": "bge-m3",
    "API_KEY": "your api key",
    
    # 本地模型路径
    "LOCAL_MODEL_PATH": r"D:\Models\bge-small-zh-v1.5",
    
    "APP_TITLE": "DRS Search Lab - 检索质量验证平台 (SQLite版)"
}

# ==========================================
# 1. 核心检索引擎 (不依赖 GUI)
# ==========================================
class VectorSearchEngine:
    """
    负责内存中的向量计算
    逻辑：加载 SQLite -> 转 Numpy 矩阵 -> 计算 Cosine Similarity
    """
    def __init__(self):
        self.metadata = []  # 存储 id, source, text
        self.matrix = None  # numpy 矩阵 (N, D)
        self.dim = 0
        self.count = 0
        self.source_model = "Unknown"

    def load_from_db(self, db_path: str, progress_callback=None):
        """从 SQLite 加载数据并构建矩阵"""
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # 获取总数用于进度条
        c.execute("SELECT COUNT(*) FROM knowledge_base")
        total = c.fetchone()[0]
        
        c.execute("SELECT id, source, text, vector, model FROM knowledge_base")
        
        vectors_list = []
        self.metadata = []
        
        fetched = 0
        batch_size = 1000
        
        while True:
            rows = c.fetchmany(batch_size)
            if not rows: break
            
            for row in rows:
                rid, source, text, vec_json, model_name = row
                try:
                    vec = json.loads(vec_json)
                    vectors_list.append(vec)
                    self.metadata.append({
                        "id": rid,
                        "source": source,
                        "text": text,
                        "model": model_name
                    })
                    self.source_model = model_name # 记录最后一条的模型用于提示
                except Exception:
                    continue
            
            fetched += len(rows)
            if progress_callback:
                progress_callback(int(fetched / total * 100))

        conn.close()
        
        if vectors_list:
            # 转换为 Numpy 矩阵，极大提升计算速度
            self.matrix = np.array(vectors_list, dtype=np.float32)
            # 归一化矩阵 (为了后续直接用点积计算余弦相似度)
            norm = np.linalg.norm(self.matrix, axis=1, keepdims=True)
            self.matrix = self.matrix / (norm + 1e-10)
            
            self.count, self.dim = self.matrix.shape
            return True, f"加载成功: {self.count} 条向量 (维度: {self.dim}, 源模型: {self.source_model})"
        else:
            return False, "数据库为空或解析失败"

    def search(self, query_vec: List[float], top_k: int = 5) -> List[Dict]:
        """执行矩阵运算检索"""
        if self.matrix is None: return []

        # 1. 查询向量归一化
        q_vec = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        q_vec = q_vec / (q_norm + 1e-10)

        # 2. 矩阵点积 (Cosine Similarity)
        # (N, D) dot (D,) -> (N,)
        scores = np.dot(self.matrix, q_vec)

        # 3. 获取 Top-K 索引
        # argsort 返回的是从小到大的索引，所以取反切片
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            meta = self.metadata[idx]
            results.append({
                "score": score,
                "source": meta["source"],
                "text": meta["text"],
                "id": meta["id"]
            })
        
        return results

# ==========================================
# 2. 异步工作线程
# ==========================================

class LoadDBWorker(QThread):
    """异步加载数据库"""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, engine, db_path):
        super().__init__()
        self.engine = engine
        self.db_path = db_path

    def run(self):
        try:
            success, msg = self.engine.load_from_db(self.db_path, self.progress_signal.emit)
            self.finished_signal.emit(success, msg)
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class SearchWorker(QThread):
    """
    负责：
    1. 生成查询的 Embedding (本地/远程)
    2. 调用 Engine 进行矩阵检索
    """
    result_signal = pyqtSignal(list, float) # results, time_cost
    log_signal = pyqtSignal(str)

    def __init__(self, engine, query_text, mode, local_model, top_k):
        super().__init__()
        self.engine = engine
        self.text = query_text
        self.mode = mode
        self.local_model = local_model
        self.top_k = top_k

    def run(self):
        t_start = time.time()
        query_vec = []

        try:
            # Step 1: Vectorize Query
            if self.mode == "local":
                if not self.local_model:
                    self.log_signal.emit("❌ 本地模型未加载")
                    return
                self.log_signal.emit("🧠 [Local] 正在计算查询向量...")
                query_vec = self.local_model.encode(self.text, normalize_embeddings=True).tolist()
            
            elif self.mode == "remote":
                self.log_signal.emit("☁️ [Remote] 正在请求 API 向量化...")
                headers = {
                    "Authorization": f"Bearer {GLOBAL_CONFIG['API_KEY']}", 
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": GLOBAL_CONFIG['REMOTE_MODEL_NAME'],
                    "input": [self.text]
                }
                resp = requests.post(GLOBAL_CONFIG['EMBED_API_URL'], json=payload, headers=headers, verify=False, timeout=10)
                if resp.status_code == 200:
                    query_vec = resp.json()['data'][0]['embedding']
                else:
                    self.log_signal.emit(f"❌ API 错误: {resp.text}")
                    return

            # Step 2: Matrix Search
            self.log_signal.emit("🔍 [Matrix] 正在执行矩阵运算...")
            results = self.engine.search(query_vec, self.top_k)
            
            t_cost = time.time() - t_start
            self.result_signal.emit(results, t_cost)

        except Exception as e:
            self.log_signal.emit(f"❌ 检索异常: {traceback.format_exc()}")

class ModelInitWorker(QThread):
    """预加载本地模型"""
    finished_signal = pyqtSignal(object)
    
    def run(self):
        try:
            if os.path.exists(GLOBAL_CONFIG['LOCAL_MODEL_PATH']):
                model = SentenceTransformer(GLOBAL_CONFIG['LOCAL_MODEL_PATH'], device='cpu')
                self.finished_signal.emit(model)
            else:
                self.finished_signal.emit(None)
        except:
            self.finished_signal.emit(None)

# ==========================================
# 3. GUI 主界面
# ==========================================

class SearchLabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = VectorSearchEngine()
        self.local_model = None
        self.init_ui()
        
        # 自动加载本地模型
        self.status_lbl.setText("⏳ 正在后台预加载本地模型...")
        self.loader = ModelInitWorker()
        self.loader.finished_signal.connect(self.on_model_ready)
        self.loader.start()

    def on_model_ready(self, model):
        self.local_model = model
        status = "✅ 本地模型就绪" if model else "⚠️ 本地模型未找到 (仅远程可用)"
        self.status_lbl.setText(status)
        if model:
            self.combo_mode.setCurrentIndex(0) # 默认切到 Local

    def init_ui(self):
        self.setWindowTitle(GLOBAL_CONFIG['APP_TITLE'])
        self.resize(1200, 800)
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Top: 数据源与配置 ---
        grp_cfg = QGroupBox("1. 数据源与引擎配置")
        grp_cfg.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        
        layout_cfg = QHBoxLayout()
        
        # DB 选择
        self.btn_db = QPushButton("📂 加载 SQLite 数据库 (.db)")
        self.btn_db.clicked.connect(self.load_db)
        self.lbl_db_info = QLabel("未加载数据")
        self.lbl_db_info.setStyleSheet("color: #666;")
        
        # 模式选择
        layout_cfg.addWidget(self.btn_db)
        layout_cfg.addWidget(self.lbl_db_info)
        layout_cfg.addStretch()
        
        layout_cfg.addWidget(QLabel("查询编码模式:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("🚀 Local (BGE-Small)", "local")
        self.combo_mode.addItem("☁️ Remote (BGE-M3)", "remote")
        layout_cfg.addWidget(self.combo_mode)
        
        grp_cfg.setLayout(layout_cfg)
        layout.addWidget(grp_cfg)

        # --- Mid: 检索交互区 ---
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Search & List
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0,0,0,0)
        
        # Search Bar
        search_bar = QHBoxLayout()
        self.input_query = QTextEdit()
        self.input_query.setPlaceholderText("在此输入业务问题，例如：'如何处理设备过热故障？' (按 Ctrl+Enter 搜索)")
        self.input_query.setFixedHeight(60)
        self.input_query.setStyleSheet("font-size: 14px;")
        
        # 绑定 Ctrl+Enter
        # (简单起见，用 EventFilter 或按钮)
        
        btn_search = QPushButton("🔍 立即检索")
        btn_search.setFixedHeight(60)
        btn_search.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; font-size: 14px;")
        btn_search.clicked.connect(self.do_search)
        
        search_bar.addWidget(self.input_query)
        search_bar.addWidget(btn_search)
        left_layout.addLayout(search_bar)
        
        # Params
        param_bar = QHBoxLayout()
        param_bar.addWidget(QLabel("Top-K (召回数):"))
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 50)
        self.spin_topk.setValue(5)
        param_bar.addWidget(self.spin_topk)
        
        param_bar.addWidget(QLabel("   阈值过滤 (Visual Only):"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.0, 1.0)
        self.spin_thresh.setSingleStep(0.05)
        self.spin_thresh.setValue(0.4)
        param_bar.addWidget(self.spin_thresh)
        
        param_bar.addStretch()
        left_layout.addLayout(param_bar)
        
        # Result Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Score", "Source File", "Snippet Preview"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemClicked.connect(self.on_table_click)
        left_layout.addWidget(self.table)
        
        splitter.addWidget(left_widget)
        
        # Right: Details
        right_grp = QGroupBox("📄 召回片段详情 (Hit Context)")
        right_layout = QVBoxLayout()
        self.txt_detail = QTextEdit()
        self.txt_detail.setReadOnly(True)
        self.txt_detail.setStyleSheet("font-family: Consolas; font-size: 12px; background-color: #f5f5f5;")
        right_layout.addWidget(self.txt_detail)
        right_grp.setLayout(right_layout)
        splitter.addWidget(right_grp)
        
        splitter.setSizes([700, 500])
        layout.addWidget(splitter)

        # --- Bottom: Status ---
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("border-top: 1px solid #ccc; padding: 5px; color: #333;")
        layout.addWidget(self.status_lbl)

    # ==========================
    # Logic
    # ==========================

    def load_db(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 SQLite 数据库", "", "DB Files (*.db);;All Files (*.*)")
        if not path: return
        
        self.btn_db.setEnabled(False)
        self.status_lbl.setText(f"正在加载 {os.path.basename(path)}，这可能需要几秒钟...")
        
        self.db_worker = LoadDBWorker(self.engine, path)
        self.db_worker.finished_signal.connect(self.on_db_loaded)
        self.db_worker.start()

    def on_db_loaded(self, success, msg):
        self.btn_db.setEnabled(True)
        if success:
            self.lbl_db_info.setText(msg)
            self.lbl_db_info.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(self, "加载完成", msg)
            
            # 智能提示模型匹配
            db_model = self.engine.source_model
            if "bge-m3" in db_model.lower():
                self.combo_mode.setCurrentIndex(1) # 切到 Remote
                QMessageBox.information(self, "建议", f"检测到数据库使用模型 '{db_model}'，\n已自动切换为 Remote 模式以匹配精度。")
            else:
                self.combo_mode.setCurrentIndex(0) # 切到 Local
        else:
            QMessageBox.critical(self, "加载失败", msg)

    def do_search(self):
        query = self.input_query.toPlainText().strip()
        if not query:
            return
        
        if self.engine.matrix is None:
            QMessageBox.warning(self, "警告", "请先加载数据库！")
            return

        mode = self.combo_mode.currentData()
        top_k = self.spin_topk.value()
        
        self.status_lbl.setText(f"🔍 Searching ({mode})...")
        self.table.setRowCount(0)
        self.txt_detail.clear()
        
        self.search_worker = SearchWorker(self.engine, query, mode, self.local_model, top_k)
        self.search_worker.result_signal.connect(self.on_search_done)
        self.search_worker.log_signal.connect(lambda s: self.status_lbl.setText(s))
        self.search_worker.start()

    def on_search_done(self, results, time_cost):
        self.status_lbl.setText(f"✅ 检索完成，耗时 {time_cost:.4f}s")
        
        self.table.setRowCount(len(results))
        thresh = self.spin_thresh.value()
        
        for i, item in enumerate(results):
            score = item['score']
            
            # 1. Score Item
            score_item = QTableWidgetItem(f"{score:.4f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            # Color coding
            if score >= 0.7:
                score_item.setBackground(QColor("#d4f7d4")) # Green
            elif score < 0.4:
                score_item.setBackground(QColor("#fadbd8")) # Red
            elif score < thresh:
                score_item.setForeground(QColor("gray")) # Grey out low scores
            
            self.table.setItem(i, 0, score_item)
            
            # 2. Source Item
            self.table.setItem(i, 1, QTableWidgetItem(item['source']))
            
            # 3. Text Item
            txt_preview = item['text'].replace("\n", " ")[:80] + "..."
            self.table.setItem(i, 2, QTableWidgetItem(txt_preview))
            
            # Store full data
            self.table.item(i, 0).setData(Qt.UserRole, item)

    def on_table_click(self, item):
        row = item.row()
        data_item = self.table.item(row, 0).data(Qt.UserRole)
        
        full_text = (
            f"=== 🔍 命中详情 (Rank {row+1}) ===\n"
            f"分数: {data_item['score']:.6f}\n"
            f"来源: {data_item['source']}\n"
            f"数据库ID: {data_item['id']}\n"
            f"----------------------------------\n"
            f"{data_item['text']}\n"
        )
        self.txt_detail.setText(full_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    win = SearchLabWindow()
    win.show()

    sys.exit(app.exec_())
