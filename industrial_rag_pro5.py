import sys
import os
import json
import sqlite3
import time
import requests
import urllib3
import traceback
import math
import random
import numpy as np
from typing import List, Dict, Tuple

# Disable urllib3 security warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. Dependencies & Global Config
# ==========================================
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                                 QFileDialog, QSplitter, QMessageBox, QProgressBar, 
                                 QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, 
                                 QAbstractItemView, QComboBox, QDoubleSpinBox, QSpinBox,
                                 QStyleFactory, QCheckBox, QTabWidget, QTextBrowser, 
                                 QGridLayout, QSlider, QFrame)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QSize, QTimer, QPointF
    from PyQt5.QtGui import QFont, QColor, QIcon, QTextCursor, QPainter, QPen, QBrush, QPalette
    from sentence_transformers import SentenceTransformer, util
except ImportError as e:
    print("【Error】Missing libraries. Please run: pip install PyQt5 sentence-transformers numpy requests")
    sys.exit(1)

GLOBAL_CONFIG = {
    "BASE_URL": "https://www.deepseek.com:18080/v1",
    "API_KEY": "your api key",
    "MODEL_EMBED": "bge-m3",
    "MODEL_RERANK": "bge-reranker-v2-m3",
    "MODEL_REWRITE_DEFAULT": "DeepSeek-V3",
    "MODEL_GEN_DEFAULT": "DeepSeek-R1",
    "LOCAL_MODEL_PATH": r"D:\Models\bge-small-zh-v1.5",
    "APP_TITLE": "Industrial RAG Pro - Enterprise Knowledge Platform"
}

# ==========================================
# 1. Network Client
# ==========================================
class APIClient:
    @staticmethod
    def post(endpoint, payload, timeout=60):
        url = f"{GLOBAL_CONFIG['BASE_URL']}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {GLOBAL_CONFIG['API_KEY']}",
            "Content-Type": "application/json"
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, verify=False, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.json()
            else:
                return False, f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            return False, f"Network Error: {str(e)}"

# ==========================================
# 2. Vector Search Engine
# ==========================================
class VectorSearchEngine:
    def __init__(self):
        self.metadata = []
        self.matrix = None
        self.dim = 0
        self.count = 0
        self.source_model = "Unknown"

    def load_from_db(self, db_path: str, progress_callback=None):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        try:
            c.execute("SELECT COUNT(*) FROM chunks_full_index")
            table_name = "chunks_full_index"
            query = "SELECT rowid, doc_title, pure_text, vector_blob FROM chunks_full_index"
            is_blob = True
        except:
            try:
                c.execute("SELECT COUNT(*) FROM knowledge_base")
                table_name = "knowledge_base"
                query = "SELECT id, source, text, vector, model FROM knowledge_base"
                is_blob = False
            except:
                conn.close()
                return False, "Unrecognized DB Schema (needs chunks_full_index or knowledge_base)"

        c.execute(f"SELECT COUNT(*) FROM {table_name}")
        total = c.fetchone()[0]
        c.execute(query)
        
        vectors_list = []
        self.metadata = []
        fetched = 0
        batch_size = 1000
        
        while True:
            rows = c.fetchmany(batch_size)
            if not rows: break
            
            for row in rows:
                try:
                    if is_blob:
                        rid, source, text, vec_data = row
                        vec = np.frombuffer(vec_data, dtype=np.float32)
                        model_name = "BlobData"
                    else:
                        rid, source, text, vec_json, model_name = row
                        vec = json.loads(vec_json)
                    
                    vectors_list.append(vec)
                    self.metadata.append({
                        "id": rid,
                        "source": source,
                        "text": text,
                        "model": model_name
                    })
                except Exception:
                    continue
            
            fetched += len(rows)
            if progress_callback:
                progress_callback(int(fetched / total * 100))

        conn.close()
        
        if vectors_list:
            self.matrix = np.array(vectors_list, dtype=np.float32)
            norm = np.linalg.norm(self.matrix, axis=1, keepdims=True)
            self.matrix = self.matrix / (norm + 1e-10)
            self.count, self.dim = self.matrix.shape
            self.source_model = self.metadata[0]['model']
            return True, f"Loaded: {self.count} vectors (Dim: {self.dim})"
        else:
            return False, "Database is empty"

    def search(self, query_vec: List[float], top_k: int) -> List[Dict]:
        if self.matrix is None: return []
        q_vec = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        q_vec = q_vec / (q_norm + 1e-10)

        scores = np.dot(self.matrix, q_vec)
        effective_k = min(top_k, self.count)
        top_indices = np.argsort(scores)[-effective_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "rerank_score": float(scores[idx]),
                "source": self.metadata[idx]["source"],
                "text": self.metadata[idx]["text"],
                "id": self.metadata[idx]["id"],
                "stage": "Vector"
            })
        return results

# ==========================================
# 3. Workers
# ==========================================

class LoadDBWorker(QThread):
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

class RAGPipelineWorker(QThread):
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.engine = params['engine']
        self.logs = []

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        full_msg = f"[{ts}] {msg}"
        self.logs.append(full_msg)
        self.log_signal.emit(full_msg)

    def run(self):
        final_pack = {
            "query_raw": self.params['query'],
            "query_rewrite": "",
            "docs": [],
            "answer": "",
            "logs": ""
        }

        try:
            # === Step 1: Query Rewrite ===
            current_q = self.params['query']
            if self.params['enable_rewrite']:
                self._log(f"🔄 Rewriting query via {self.params['model_rewrite']}...")
                rw_payload = {
                    "model": self.params['model_rewrite'],
                    "messages": [{"role": "user", "content": f"Rewrite this query for knowledge retrieval, output only the rewritten query: {current_q}"}],
                    "temperature": 0.3
                }
                ok, res = APIClient.post("chat/completions", rw_payload)
                if ok and 'choices' in res:
                    rewritten = res['choices'][0]['message']['content'].strip()
                    self._log(f"✅ Rewritten: {current_q} -> {rewritten}")
                    current_q = rewritten
                    final_pack['query_rewrite'] = rewritten
                else:
                    self._log(f"⚠️ Rewrite failed, using original.")
            
            # === Step 2: Vector Search ===
            self._log(f"🔎 Vector Recall (Top {self.params['top_k_recall']})...")
            
            # 2.1 Embedding
            emb_payload = {"model": GLOBAL_CONFIG['MODEL_EMBED'], "input": [current_q]}
            ok, res = APIClient.post("embeddings", emb_payload)
            if not ok: raise Exception(f"Embedding API Error: {res}")
            
            q_vec = res['data'][0]['embedding']
            
            # 2.2 Matrix Search
            candidates = self.engine.search(q_vec, top_k=self.params['top_k_recall'])
            self._log(f"✅ Recall complete. Candidates: {len(candidates)}. Top Score: {candidates[0]['score']:.4f}" if candidates else "⚠️ No results")

            # === Step 3: Reranking ===
            # 3.1 Local Filter
            if self.params['local_model'] and self.params['use_local_filter']:
                self._log("🧠 Local BGE-Small Filter...")
                texts = [c['text'] for c in candidates]
                l_q_vec = self.params['local_model'].encode(current_q, convert_to_tensor=True)
                l_d_vecs = self.params['local_model'].encode(texts, convert_to_tensor=True)
                l_scores = util.cos_sim(l_q_vec, l_d_vecs)[0]
                
                for i, score in enumerate(l_scores):
                    candidates[i]['local_score'] = float(score)
                candidates.sort(key=lambda x: x.get('local_score', 0), reverse=True)
                # Apply recall limit again after local sort before sending to remote
                candidates = candidates[:self.params['top_k_recall']] 

            # 3.2 Remote Rerank
            rerank_k = self.params['top_k_rerank']
            self._log(f"⚖️ Remote Rerank (Top {rerank_k})...")
            
            rerank_payload = {
                "model": GLOBAL_CONFIG['MODEL_RERANK'],
                "query": current_q,
                "documents": [c['text'] for c in candidates],
                "top_n": len(candidates) # Send all for scoring, cut later
            }
            ok, res = APIClient.post("rerank", rerank_payload)
            
            final_docs = []
            if ok:
                rank_results = res.get('results', [])
                for item in rank_results:
                    idx = item['index']
                    doc = candidates[idx]
                    doc['rerank_score'] = item['relevance_score']
                    doc['stage'] = "Reranked"
                    final_docs.append(doc)
                final_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
                # Cut to Rerank Top K
                final_docs = final_docs[:rerank_k]
                self._log(f"✅ Rerank complete. Top Score: {final_docs[0]['rerank_score']:.4f}")
            else:
                self._log(f"❌ Rerank API failed, using vector scores.")
                final_docs = candidates[:rerank_k]

            final_pack['docs'] = final_docs

            # === Step 4: Generation ===
            self._log(f"💬 Generating answer ({self.params['model_gen']})...")
            
            # Use top 5 for context window
            ctx_str = "\n\n".join([f"[Ref {i+1}] {d['text']}" for i, d in enumerate(final_docs[:5])])
            prompt = (
                f"You are an enterprise knowledge assistant. Answer based on the references.\n"
                f"---References---\n{ctx_str}\n----------------\n\n"
                f"Question: {self.params['query']}"
            )
            
            gen_payload = {
                "model": self.params['model_gen'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5
            }
            
            ok, res = APIClient.post("chat/completions", gen_payload, timeout=120)
            if ok and 'choices' in res:
                ans = res['choices'][0]['message']['content']
                final_pack['answer'] = ans
                self._log("✨ Generation complete.")
            else:
                final_pack['answer'] = f"Generation Failed: {res}"
                self._log("❌ Generation Failed.")

        except Exception as e:
            err = traceback.format_exc()
            self._log(f"💥 Exception: {err}")
            final_pack['answer'] = f"Error:\n{err}"

        final_pack['logs'] = "\n".join(self.logs)
        self.result_signal.emit(final_pack)

class ModelInitWorker(QThread):
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
# 4. Custom UI Components (Particles)
# ==========================================

class ParticleCanvas(QWidget):
    """Google-style particle background"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.particles = []
        self.num_particles = 45
        self.connect_dist = 140
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(35) # ~30 FPS
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True) # Let clicks pass through if needed

    def resizeEvent(self, event):
        self.init_particles(event.size().width(), event.size().height())
        super().resizeEvent(event)

    def init_particles(self, w, h):
        self.particles = []
        for _ in range(self.num_particles):
            self.particles.append({
                'x': random.uniform(0, w),
                'y': random.uniform(0, h),
                'vx': random.uniform(-0.6, 0.6),
                'vy': random.uniform(-0.6, 0.6)
            })

    def update_animation(self):
        w = self.width()
        h = self.height()
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            
            # Bounce
            if p['x'] <= 0 or p['x'] >= w: p['vx'] *= -1
            if p['y'] <= 0 or p['y'] >= h: p['vy'] *= -1

        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background color (Simulate #F0F2F5)
        painter.fillRect(self.rect(), QColor("#F0F2F5"))

        # Draw particles
        pen_node = QPen(QColor("#4285F4"))
        pen_node.setWidth(0)
        brush_node = QBrush(QColor("#4285F4"))
        
        for i, p in enumerate(self.particles):
            # Draw node
            painter.setPen(pen_node)
            painter.setBrush(brush_node)
            painter.drawEllipse(QPointF(p['x'], p['y']), 2, 2)
            
            # Draw connections
            for j in range(i + 1, len(self.particles)):
                p2 = self.particles[j]
                dx = p['x'] - p2['x']
                dy = p['y'] - p2['y']
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < self.connect_dist:
                    alpha = int(220 * (1 - dist / self.connect_dist))
                    line_color = QColor("#4285F4")
                    line_color.setAlpha(alpha)
                    painter.setPen(QPen(line_color, 1))
                    painter.drawLine(QPointF(p['x'], p['y']), QPointF(p2['x'], p2['y']))

# ==========================================
# 5. Main Window & Layout
# ==========================================

class IndustrialRAGWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = VectorSearchEngine()
        self.local_model = None
        
        # Init UI
        self.init_ui()
        
        # Load local model
        self.log_to_console("⏳ Loading local BGE model in background...")
        self.loader = ModelInitWorker()
        self.loader.finished_signal.connect(self.on_model_ready)
        self.loader.start()

    def on_model_ready(self, model):
        self.local_model = model
        status = "✅ Local model ready" if model else "⚠️ Local model not found (Local filter disabled)"
        self.log_to_console(status)
        self.chk_local_filter.setEnabled(bool(model))

    def init_ui(self):
        self.setWindowTitle(GLOBAL_CONFIG['APP_TITLE'])
        self.resize(1300, 900)
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # 1. Main Container (Stack to hold Particles + Content)
        main_container = QWidget()
        self.setCentralWidget(main_container)
        stack_layout = QGridLayout(main_container)
        stack_layout.setContentsMargins(0,0,0,0)

        # 2. Layer 0: Particle Background
        self.particles = ParticleCanvas()
        stack_layout.addWidget(self.particles, 0, 0)

        # 3. Layer 1: Content Widget (Transparent background)
        content_widget = QWidget()
        stack_layout.addWidget(content_widget, 0, 0)
        
        # Main Horizontal Layout
        main_h_layout = QHBoxLayout(content_widget)
        main_h_layout.setContentsMargins(15, 15, 15, 15)

        # === Splitter ===
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        main_h_layout.addWidget(splitter)

        # =============================
        # LEFT PANEL (2/3 Width)
        # Top: Config | Bottom: Chat
        # =============================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0,0,5,0)

        # --- A. Configuration Group ---
        config_group = QGroupBox("🛠️ Pipeline Configuration")
        config_group.setStyleSheet("QGroupBox { background-color: rgba(255,255,255,0.8); font-weight: bold; border: 1px solid #aaa; border-radius: 5px; padding-top: 15px; }")
        grid_config = QGridLayout()
        
        # Row 1: DB & Rewrite
        self.btn_db = QPushButton("📂 Mount DB")
        self.btn_db.clicked.connect(self.load_db)
        grid_config.addWidget(self.btn_db, 0, 0)
        
        self.lbl_db_path = QLabel("Unmounted")
        self.lbl_db_path.setStyleSheet("color: red; font-size: 10px;")
        grid_config.addWidget(self.lbl_db_path, 0, 1)

        grid_config.addWidget(QLabel("Rewrite Model:"), 0, 2)
        self.combo_rewrite = QComboBox()
        self.combo_rewrite.addItems([GLOBAL_CONFIG['MODEL_REWRITE_DEFAULT'], "deepseek-chat"])
        grid_config.addWidget(self.combo_rewrite, 0, 3)
        
        self.chk_rewrite = QCheckBox("Enable")
        self.chk_rewrite.setChecked(True)
        grid_config.addWidget(self.chk_rewrite, 0, 4)

        # Row 2: Recall & Rerank & Gen
        # Recall Control (5-80)
        recall_layout = QHBoxLayout()
        recall_layout.addWidget(QLabel("Recall Top-K:"))
        self.spin_recall = QSpinBox()
        self.spin_recall.setRange(5, 80)
        self.spin_recall.setValue(50)
        recall_layout.addWidget(self.spin_recall)
        grid_config.addLayout(recall_layout, 1, 0, 1, 2)

        # Rerank Control (5-50)
        rerank_layout = QHBoxLayout()
        rerank_layout.addWidget(QLabel("Rerank Top-K:"))
        self.spin_rerank = QSpinBox()
        self.spin_rerank.setRange(5, 50)
        self.spin_rerank.setValue(30)
        rerank_layout.addWidget(self.spin_rerank)
        grid_config.addLayout(rerank_layout, 1, 2, 1, 2)

        # Local Filter Checkbox
        self.chk_local_filter = QCheckBox("Local Filter")
        self.chk_local_filter.setChecked(True)
        grid_config.addWidget(self.chk_local_filter, 1, 4)

        # Row 3: Gen Model
        grid_config.addWidget(QLabel("Gen Model:"), 2, 0)
        self.combo_gen = QComboBox()
        self.combo_gen.addItems([GLOBAL_CONFIG['MODEL_GEN_DEFAULT'], "DeepSeek-V3"])
        grid_config.addWidget(self.combo_gen, 2, 1, 1, 2)

        config_group.setLayout(grid_config)
        left_layout.addWidget(config_group)

        # --- B. Chat Display Area ---
        self.browser_chat = QTextBrowser()
        self.browser_chat.setOpenExternalLinks(True)
        # Semi-transparent background for chat
        self.browser_chat.setStyleSheet("""
            QTextBrowser { 
                background-color: rgba(255, 255, 255, 0.9); 
                border: 1px solid #ccc; 
                border-radius: 5px;
                font-family: 'Segoe UI', sans-serif; 
                font-size: 14px; 
                padding: 10px;
            }
        """)
        self.browser_chat.setHtml("<h3 style='color:#555;'>💡 Ready to serve.</h3><p>Results will appear here.</p>")
        left_layout.addWidget(self.browser_chat)
        
        splitter.addWidget(left_widget)

        # =============================
        # RIGHT PANEL (1/3 Width)
        # Top: Input | Bottom: Logs/Evidence
        # =============================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5,0,0,0)

        # --- C. Input Area (Top) ---
        input_group = QGroupBox("User Input (Prompt)")
        input_group.setStyleSheet("QGroupBox { background-color: rgba(255,255,255,0.8); font-weight: bold; border: 1px solid #aaa; border-radius: 5px; padding-top: 10px; }")
        input_vbox = QVBoxLayout()
        
        self.input_query = QTextEdit()
        self.input_query.setPlaceholderText("Enter your business question here...")
        self.input_query.setFixedHeight(100)
        input_vbox.addWidget(self.input_query)

        self.btn_send = QPushButton("⚡ Execute Pipeline")
        self.btn_send.setFixedHeight(40)
        self.btn_send.setStyleSheet("background-color: #4285F4; color: white; font-weight: bold; font-size: 14px; border-radius: 3px;")
        self.btn_send.clicked.connect(self.do_search)
        input_vbox.addWidget(self.btn_send)

        input_group.setLayout(input_vbox)
        right_layout.addWidget(input_group)

        # --- D. Tabs: Logs (Default) & Docs ---
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #aaa; background: rgba(255,255,255,0.9); }
            QTabBar::tab { background: #ddd; padding: 5px 10px; border: 1px solid #999; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #fff; font-weight: bold; }
        """)

        # Tab 1: System Logs (Default)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 10px;")
        self.tabs.addTab(self.txt_log, "💻 System Logs")

        # Tab 2: Evidence Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Score", "Source", "Preview"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemClicked.connect(self.on_table_click)
        self.table.setStyleSheet("background-color: rgba(255,255,255,0.9);")
        self.tabs.addTab(self.table, "📄 Evidence Docs")

        self.tabs.setCurrentIndex(0) # Default to Logs
        right_layout.addWidget(self.tabs)

        # --- E. Transparency Slider ---
        trans_layout = QHBoxLayout()
        trans_layout.addWidget(QLabel("Opacity:"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(40, 100)
        self.slider_alpha.setValue(100)
        self.slider_alpha.valueChanged.connect(self.change_opacity)
        trans_layout.addWidget(self.slider_alpha)
        
        # Add a container for slider to have background
        slider_container = QWidget()
        slider_container.setStyleSheet("background-color: rgba(255,255,255,0.7); border-radius: 5px;")
        slider_container.setLayout(trans_layout)
        right_layout.addWidget(slider_container)

        splitter.addWidget(right_widget)
        
        # Set Splitter Ratios (2/3 Left, 1/3 Right)
        splitter.setSizes([850, 450])

    # ==========================
    # Logic Implementation
    # ==========================

    def change_opacity(self, value):
        self.setWindowOpacity(value / 100.0)

    def log_to_console(self, msg):
        self.txt_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        cursor = self.txt_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.txt_log.setTextCursor(cursor)

    def load_db(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select DB", "", "SQLite DB (*.db)")
        if not path: return
        
        self.btn_db.setEnabled(False)
        self.log_to_console("📂 Mounting database...")
        self.db_worker = LoadDBWorker(self.engine, path)
        self.db_worker.finished_signal.connect(self.on_db_loaded)
        self.db_worker.start()

    def on_db_loaded(self, success, msg):
        self.btn_db.setEnabled(True)
        if success:
            self.lbl_db_path.setText(os.path.basename(self.db_worker.db_path))
            self.lbl_db_path.setStyleSheet("color: green; font-weight: bold;")
            self.log_to_console(f"✅ {msg}")
        else:
            self.lbl_db_path.setText("Error")
            self.log_to_console(f"❌ {msg}")
            QMessageBox.critical(self, "Error", msg)

    def do_search(self):
        query = self.input_query.toPlainText().strip()
        if not query: return
        if self.engine.matrix is None:
            QMessageBox.warning(self, "Warning", "Please mount a database first.")
            return

        self.btn_send.setEnabled(False)
        self.table.setRowCount(0)
        self.txt_log.clear()
        self.browser_chat.setHtml(f"<h3>⏳ Processing...</h3><p>Query: {query}</p>")
        
        params = {
            "engine": self.engine,
            "query": query,
            "enable_rewrite": self.chk_rewrite.isChecked(),
            "model_rewrite": self.combo_rewrite.currentText(),
            "model_gen": self.combo_gen.currentText(),
            "top_k_recall": self.spin_recall.value(),
            "top_k_rerank": self.spin_rerank.value(),
            "use_local_filter": self.chk_local_filter.isChecked(),
            "local_model": self.local_model,
        }

        self.worker = RAGPipelineWorker(params)
        self.worker.log_signal.connect(self.log_to_console)
        self.worker.result_signal.connect(self.on_pipeline_finish)
        self.worker.start()

    def on_pipeline_finish(self, result):
        self.btn_send.setEnabled(True)
        self.log_to_console("🏁 Pipeline Finished.")
        
        # 1. Update Chat
        html = f"""
        <div style="font-family: sans-serif;">
            <div style="background-color:#e6f3ff; padding:8px; border-radius:4px; margin-bottom:8px;">
                <b>❓ Query:</b> {result['query_raw']}
            </div>
        """
        if result['query_rewrite']:
            html += f"""
            <div style="background-color:#fff0e6; padding:8px; border-radius:4px; margin-bottom:8px;">
                <b>🔄 Rewrite:</b> {result['query_rewrite']}
            </div>
            """
        
        html += f"""
            <div style="margin-top:15px;">
                <b>🤖 Answer ({self.combo_gen.currentText()}):</b><br>
                <hr style='border: 0; height: 1px; background: #ccc;'>
                <div style="font-size:15px; line-height:1.5;">{result['answer'].replace(chr(10), '<br>')}</div>
            </div>
        </div>
        """
        self.browser_chat.setHtml(html)

        # 2. Update Table
        docs = result['docs']
        self.table.setRowCount(len(docs))
        for i, doc in enumerate(docs):
            # Score
            score_val = doc.get('rerank_score', doc.get('score', 0))
            item_score = QTableWidgetItem(f"{score_val:.4f}")
            if score_val > 0.7: item_score.setBackground(QColor("#d4f7d4"))
            elif score_val < 0.2: item_score.setBackground(QColor("#fadbd8"))
            self.table.setItem(i, 0, item_score)
            
            # Source
            self.table.setItem(i, 1, QTableWidgetItem(str(doc.get('source', 'Unknown'))))
            
            # Text Preview
            preview = doc['text'].replace("\n", " ")[:60] + "..."
            self.table.setItem(i, 2, QTableWidgetItem(preview))
            
            # Store data
            self.table.item(i, 0).setData(Qt.UserRole, doc)

    def on_table_click(self, item):
        row = item.row()
        data = self.table.item(row, 0).data(Qt.UserRole)
        QMessageBox.information(self, f"Doc Detail (Rank {row+1})", 
                                f"Source: {data['source']}\nID: {data['id']}\nScore: {data.get('rerank_score',0)}\n\n{data['text']}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei UI", 9)
    app.setFont(font)
    win = IndustrialRAGWindow()
    win.show()

    sys.exit(app.exec_())
