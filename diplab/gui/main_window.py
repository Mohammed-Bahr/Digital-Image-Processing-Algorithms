"""Main window: algorithm browser, live preview, parameter controls."""

from __future__ import annotations

import os
import time
import traceback

import numpy as np
from PIL import Image
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (QAbstractItemView, QAction, QApplication,
                             QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPlainTextEdit, QPushButton,
                             QScrollArea, QSizePolicy, QSpinBox, QSplitter,
                             QTabWidget, QToolBar, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from .. import frequency as fr
from ..demo import make_demo_image
from ..operations import (CATEGORY_ORDER, OPERATIONS, OpResult, Operation,
                          Param, find_operation, operations_by_category)
from ..utils import (as_uint8, describe_kernel, format_matrix,
                    histogram_counts, resize_bilinear)
from .widgets import CurveWidget, HistogramWidget, ImageView

MAX_SOURCE_SIDE = 4096    # safety cap for huge photos
WORK_SIZES = ("1024 px", "1536 px", "2048 px", "Full")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIP Lab — algorithms from the notes")
        self.resize(1480, 940)
        self.setMinimumSize(1120, 700)
        self.setAcceptDrops(True)

        # ---- state -------------------------------------------------------
        self.original_img: np.ndarray | None = None
        self.work_img: np.ndarray | None = None
        self.result_img: np.ndarray | None = None
        self.result: OpResult | None = None
        self.current_op: Operation | None = None
        self.source_name = "demo"
        self.work_limit: int | None = 1024
        self.param_widgets: dict[str, tuple[Param, QWidget]] = {}
        self.tree_items: dict[str, QTreeWidgetItem] = {}

        self.apply_timer = QTimer(self)
        self.apply_timer.setSingleShot(True)
        self.apply_timer.setInterval(180)
        self.apply_timer.timeout.connect(self._apply_now)

        self._build_actions()
        self._build_toolbar()
        self._build_menu()
        self._build_ui()
        self._populate_tree()

        self.statusBar().showMessage("Ready")
        self.status_info = QLabel("")
        self.status_info.setObjectName("statusInfo")
        self.statusBar().addPermanentWidget(self.status_info)

        self._load_demo()
        self._select_operation("equalize", scroll=False)
        self.tree.scrollToTop()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_actions(self):
        self.act_open = QAction("Open image", self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.triggered.connect(self._open_image)

        self.act_save = QAction("Save result", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self._save_result)

        self.act_demo = QAction("Load demo scene", self)
        self.act_demo.triggered.connect(lambda: self._load_demo())

        self.act_reset = QAction("Reset input", self)
        self.act_reset.setShortcut("Ctrl+R")
        self.act_reset.triggered.connect(self._reset_input)

        self.act_use = QAction("Use result as input", self)
        self.act_use.setShortcut("Ctrl+U")
        self.act_use.triggered.connect(self._use_result)

        self.act_apply = QAction("Apply", self)
        self.act_apply.setShortcut("F5")
        self.act_apply.triggered.connect(lambda: self._apply_now(force=True))

        self.act_quit = QAction("Quit", self)
        self.act_quit.setShortcut("Ctrl+Q")
        self.act_quit.triggered.connect(self.close)

        self.act_dft = QAction("DFT worked example (Lec 4)", self)
        self.act_dft.triggered.connect(self._show_dft_demo)

        self.act_about = QAction("Algorithm map", self)
        self.act_about.triggered.connect(self._show_about)

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)

        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addSeparator()
        tb.addAction(self.act_demo)
        tb.addAction(self.act_reset)
        tb.addAction(self.act_use)
        tb.addSeparator()
        tb.addAction(self.act_apply)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        tb.addWidget(QLabel("  Working size  "))
        self.size_combo = QComboBox()
        self.size_combo.addItems(WORK_SIZES)
        self.size_combo.setCurrentText("1024 px")
        self.size_combo.setToolTip("Images are processed at this maximum size. "
                                   "Raise it if you need more detail (FFT "
                                   "filters get slower).")
        self.size_combo.currentIndexChanged.connect(self._on_size_changed)
        tb.addWidget(self.size_combo)
        tb.addWidget(QLabel("  "))

        self.auto_check = QCheckBox("Auto preview")
        self.auto_check.setChecked(True)
        self.auto_check.setToolTip("Apply parameter changes automatically "
                                   "(uncheck, then press F5 / Apply)")
        self.auto_check.toggled.connect(self._on_auto_toggled)
        tb.addWidget(self.auto_check)
        tb.addWidget(QLabel("  "))

    def _build_menu(self):
        m_file = self.menuBar().addMenu("&File")
        m_file.addAction(self.act_open)
        m_file.addAction(self.act_save)
        m_file.addSeparator()
        m_file.addAction(self.act_demo)
        m_file.addAction(self.act_reset)
        m_file.addAction(self.act_use)
        m_file.addSeparator()
        m_file.addAction(self.act_quit)

        m_tools = self.menuBar().addMenu("&Tools")
        m_tools.addAction(self.act_apply)
        m_tools.addSeparator()
        m_tools.addAction(self.act_dft)
        m_tools.addAction(self.act_about)

        m_help = self.menuBar().addMenu("&Help")
        m_help.addAction(self.act_about)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # ---------------- left: algorithm tree ----------------------------
        left = QWidget()
        left.setMinimumWidth(245)
        left.setMaximumWidth(360)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 8, 12)
        lv.setSpacing(8)

        head = QLabel("OPERATIONS")
        head.setObjectName("sectionLabel")
        lv.addWidget(head)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search algorithms…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        lv.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setAnimated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self._on_tree_current)
        lv.addWidget(self.tree, 1)

        hint = QLabel("Pick an algorithm and adjust the parameters.\n"
                      "The result updates live on the right.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        splitter.addWidget(left)

        # ---------------- centre: images + plots --------------------------
        centre = QSplitter(Qt.Vertical)
        centre.setChildrenCollapsible(False)

        views = QSplitter(Qt.Horizontal)
        views.setChildrenCollapsible(False)
        input_pane, self.input_view, self.input_dims = self._image_pane(
            "INPUT", "No image loaded")
        result_pane, self.result_view, self.result_dims = self._image_pane(
            "RESULT", "Choose an algorithm on the left")
        views.addWidget(input_pane)
        views.addWidget(result_pane)
        views.setSizes([500, 500])
        centre.addWidget(views)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.hist_widget = HistogramWidget()
        self.curve_widget = CurveWidget()
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Monospace", 9))
        self.log_widget.setPlainText("Algorithm explanation will appear here.")
        self.tabs.addTab(self.hist_widget, "Histogram")
        self.tabs.addTab(self.curve_widget, "Transfer curve")
        self.tabs.addTab(self.log_widget, "Explanation")
        centre.addWidget(self.tabs)
        centre.setSizes([560, 330])
        splitter.addWidget(centre)

        # ---------------- right: parameters -------------------------------
        right = QWidget()
        right.setMinimumWidth(315)
        right.setMaximumWidth(410)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 12, 12, 12)
        rv.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        hv = QVBoxLayout(host)
        hv.setContentsMargins(2, 2, 6, 2)
        hv.setSpacing(8)

        self.op_title = QLabel("Choose an operation")
        self.op_title.setObjectName("opTitle")
        self.op_title.setWordWrap(True)
        self.op_meta = QLabel("")
        self.op_meta.setObjectName("opMeta")
        self.op_desc = QLabel("Every algorithm is implemented from scratch. "
                              "Pick one on the left to begin.")
        self.op_desc.setObjectName("opDesc")
        self.op_desc.setWordWrap(True)

        hv.addWidget(self.op_title)
        hv.addWidget(self.op_meta)
        hv.addWidget(self.op_desc)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("separator")
        hv.addWidget(line)

        params_label = QLabel("PARAMETERS")
        params_label.setObjectName("sectionLabel")
        hv.addWidget(params_label)

        self.params_box = QWidget()
        self.form = QFormLayout(self.params_box)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(8)
        self.form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        hv.addWidget(self.params_box)

        self.kernel_label = QLabel("")
        self.kernel_label.setObjectName("kernel")
        self.kernel_label.setWordWrap(True)
        self.kernel_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.kernel_label.hide()
        hv.addWidget(self.kernel_label)

        hv.addStretch(1)
        scroll.setWidget(host)
        rv.addWidget(scroll, 1)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.setMinimumHeight(34)
        self.apply_btn.clicked.connect(lambda: self._apply_now(force=True))
        self.use_btn = QPushButton("Use result as input")
        self.use_btn.setMinimumHeight(30)
        self.use_btn.clicked.connect(self._use_result)
        rv.addWidget(self.apply_btn)
        rv.addWidget(self.use_btn)

        self.preview_hint = QLabel("Auto preview is on.")
        self.preview_hint.setObjectName("hintLabel")
        self.preview_hint.setWordWrap(True)
        rv.addWidget(self.preview_hint)

        splitter.addWidget(right)
        splitter.setSizes([260, 880, 340])

    def _image_pane(self, title: str, placeholder: str):
        pane = QWidget()
        v = QVBoxLayout(pane)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        head = QWidget()
        h = QHBoxLayout(head)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(6)
        t = QLabel(title)
        t.setObjectName("paneTitle")
        dims = QLabel("")
        dims.setObjectName("paneDims")
        dims.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(t)
        h.addStretch(1)
        h.addWidget(dims)

        view = ImageView(placeholder)
        v.addWidget(head)
        v.addWidget(view, 1)
        return pane, view, dims

    # ------------------------------------------------------------------
    # algorithm tree
    # ------------------------------------------------------------------
    def _populate_tree(self):
        grouped = operations_by_category()
        cat_font = QFont()
        cat_font.setBold(True)

        for category in CATEGORY_ORDER:
            ops = grouped.get(category, [])
            if not ops:
                continue
            cat_item = QTreeWidgetItem([category])
            cat_item.setFont(0, cat_font)
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(cat_item)
            for op in ops:
                child = QTreeWidgetItem([op.name])
                child.setData(0, Qt.UserRole, op.key)
                child.setToolTip(0, op.description)
                cat_item.addChild(child)
                self.tree_items[op.key] = child
            cat_item.setExpanded(True)

    def _filter_tree(self, text: str):
        text = (text or "").strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            cat = self.tree.topLevelItem(i)
            visible = 0
            for j in range(cat.childCount()):
                child = cat.child(j)
                op = find_operation(child.data(0, Qt.UserRole))
                hay = (child.text(0) + " " +
                       (op.description if op else "")).lower()
                match = (not text) or (text in hay)
                child.setHidden(not match)
                if match:
                    visible += 1
            cat.setHidden(visible == 0)
            cat.setExpanded((not text) or visible > 0)

    def _select_operation(self, key: str, scroll: bool = True):
        item = self.tree_items.get(key)
        if item is not None:
            self.tree.setCurrentItem(item)
            if scroll:
                self.tree.scrollToItem(item, QAbstractItemView.PositionAtCenter)

    def _on_tree_current(self, current, previous):
        if current is None:
            return
        key = current.data(0, Qt.UserRole)
        if not key:
            return
        op = find_operation(key)
        if op is None:
            return
        self.current_op = op
        self._show_operation(op)
        self._apply_now(force=True)

    # ------------------------------------------------------------------
    # parameter panel
    # ------------------------------------------------------------------
    def _show_operation(self, op: Operation):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.param_widgets.clear()

        self.op_title.setText(op.name)
        self.op_meta.setText(op.category.upper())
        self.op_desc.setText(op.description)
        self.kernel_label.hide()
        self.kernel_label.setText("")

        for p in op.params:
            widget = self._make_widget(p)
            if widget is None:
                continue
            label = QLabel(p.label)
            label.setObjectName("paramLabel")
            self.form.addRow(label, widget)
            self.param_widgets[p.key] = (p, widget)

        if not op.params:
            none_label = QLabel("This operation has no parameters.")
            none_label.setObjectName("hintLabel")
            self.form.addRow(none_label)

    def _make_widget(self, p: Param) -> QWidget | None:
        if p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.minimum), int(p.maximum))
            w.setSingleStep(int(p.step))
            w.setValue(int(p.default))
            w.valueChanged.connect(self._schedule_apply)
            w.setMinimumHeight(28)
            return w
        if p.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(int(p.decimals))
            w.setRange(float(p.minimum), float(p.maximum))
            w.setSingleStep(float(p.step))
            w.setValue(float(p.default))
            if p.suffix:
                w.setSuffix(" " + p.suffix)
            w.valueChanged.connect(self._schedule_apply)
            w.setMinimumHeight(28)
            return w
        if p.kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in p.choices])
            w.setCurrentText(str(p.default))
            w.currentIndexChanged.connect(self._schedule_apply)
            w.setMinimumHeight(28)
            return w
        if p.kind == "bool":
            w = QCheckBox("enabled")
            w.setChecked(bool(p.default))
            w.toggled.connect(self._schedule_apply)
            return w
        return None

    def _collect_params(self) -> dict:
        out = {}
        for key, (p, w) in self.param_widgets.items():
            if p.kind == "int":
                out[key] = int(w.value())
            elif p.kind == "float":
                out[key] = float(w.value())
            elif p.kind == "choice":
                out[key] = w.currentText()
            else:
                out[key] = bool(w.isChecked())
        return out

    def _schedule_apply(self, *_):
        if not self.auto_check.isChecked():
            return
        self.apply_timer.start()

    def _on_auto_toggled(self, checked: bool):
        self.preview_hint.setText(
            "Auto preview is on — changes apply immediately."
            if checked else
            "Auto preview is off — press Apply (F5) to update.")
        if checked:
            self._apply_now(force=True)

    # ------------------------------------------------------------------
    # applying an operation
    # ------------------------------------------------------------------
    def _apply_now(self, force: bool = False):
        if self.work_img is None or self.current_op is None:
            return
        if not force and not self.auto_check.isChecked():
            return

        params = self._collect_params()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        t0 = time.perf_counter()
        try:
            result = self.current_op.func(self.work_img, params)
            if not isinstance(result, OpResult):
                raise TypeError("operation did not return an OpResult")
        except Exception as exc:              # noqa: BLE001 - show to the user
            elapsed = (time.perf_counter() - t0) * 1000
            self.statusBar().showMessage(f"{type(exc).__name__}: {exc}")
            self.log_widget.setPlainText(
                f"Operation failed after {elapsed:.1f} ms\n\n"
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        elapsed = (time.perf_counter() - t0) * 1000
        self.result = result
        self.result_img = result.image

        self.result_view.set_image(result.image)
        self.result_dims.setText(self._dims_text(result.image))
        self._update_histogram()
        self._update_curve(params)
        self._update_aux(result.aux)
        self._update_kernel(result.kernel)
        self._update_log(result, params, elapsed)
        self._update_status_info()

        self.statusBar().showMessage(
            f"{self.current_op.name} applied in {elapsed:.0f} ms")

    def _update_histogram(self):
        before = histogram_counts(self.work_img) if self.work_img is not None else None
        after = histogram_counts(self.result_img) if self.result_img is not None else None
        self.hist_widget.set_data(before, after)

    def _update_curve(self, params: dict):
        op = self.current_op
        if op is None or not op.pointwise:
            self.curve_widget.set_curve(None, None)
            return
        ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
        try:
            if op.key == "log":
                c = float(params.get("c", 0))
                if c <= 0:
                    peak = float(np.max(self.work_img))
                    c = 255.0 / np.log1p(max(peak, 1.0))
                y = c * np.log1p(ramp.reshape(-1))
            else:
                y = np.asarray(op.func(ramp, params).image).reshape(-1)
            self.curve_widget.set_curve(np.arange(256), y)
        except Exception:
            self.curve_widget.set_curve(None, None)

    def _update_aux(self, aux: dict):
        while self.tabs.count() > 3:
            widget = self.tabs.widget(3)
            self.tabs.removeTab(3)
            widget.deleteLater()
        for title, arr in aux.items():
            view = ImageView()
            view.set_image(arr)
            self.tabs.addTab(view, title)

    def _update_kernel(self, kernel):
        if kernel is None:
            self.kernel_label.hide()
            return
        k = np.asarray(kernel, dtype=np.float64)
        if k.size <= 81:
            text = (f"kernel {k.shape[0]}×{k.shape[1]}, "
                    f"sum = {float(k.sum()):g}\n\n")
            text += format_matrix(k, 3)
        else:
            text = describe_kernel(k)
        self.kernel_label.setText(text)
        self.kernel_label.show()

    def _update_log(self, result: OpResult, params: dict, elapsed: float):
        op = self.current_op
        lines = [f"{op.name}   ·   {op.category}", "=" * 62, "",
                 result.log.strip(), "", "-" * 62,
                 f"time: {elapsed:.1f} ms"]
        if params:
            pretty = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"parameters: {pretty}")
        lines.append(f"input: {self._dims_text(self.work_img)}")
        lines.append(f"output: {self._dims_text(result.image)}")
        self.log_widget.setPlainText("\n".join(lines))
        self.log_widget.verticalScrollBar().setValue(0)

    def _update_status_info(self):
        if self.work_img is None:
            self.status_info.setText("")
            return
        h, w = self.work_img.shape[:2]
        mode = "RGB" if self.work_img.ndim == 3 else "gray"
        self.status_info.setText(
            f"{w} × {h} px · {mode} · {self.source_name}")

    @staticmethod
    def _dims_text(img) -> str:
        if img is None:
            return ""
        h, w = img.shape[:2]
        mode = "RGB" if img.ndim == 3 else "gray"
        return f"{w} × {h}  ({mode})"

    # ------------------------------------------------------------------
    # input / output
    # ------------------------------------------------------------------
    def _load_demo(self):
        arr = make_demo_image(512)
        self._set_input(arr, name="demo scene", note="")

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif);;"
            "All files (*)")
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        try:
            im = Image.open(path)
            im.load()
        except Exception as exc:              # noqa: BLE001
            QMessageBox.critical(self, "Cannot open image", str(exc))
            return
        arr = self._pil_to_array(im)
        h, w = arr.shape[:2]
        note = ""
        if max(h, w) > MAX_SOURCE_SIDE:
            scale = MAX_SOURCE_SIDE / float(max(h, w))
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            arr = np.array(Image.fromarray(arr).resize(new_size,
                                                       Image.LANCZOS))
            note = f"source downscaled from {w}×{h}"
        self._set_input(arr, name=os.path.basename(path), note=note)

    @staticmethod
    def _pil_to_array(im: Image.Image) -> np.ndarray:
        if im.mode == "L":
            return np.array(im, dtype=np.uint8)
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.array(im).astype(np.float64)
            lo, hi = float(a.min()), float(a.max())
            if hi > lo:
                a = (a - lo) / (hi - lo) * 255.0
            return np.clip(np.rint(a), 0, 255).astype(np.uint8)
        if im.mode == "F":
            return np.clip(np.rint(np.array(im)), 0, 255).astype(np.uint8)
        if im.mode == "1":
            return np.array(im.convert("L"), dtype=np.uint8)
        return np.array(im.convert("RGB"), dtype=np.uint8)

    def _working_copy(self) -> tuple[np.ndarray | None, str]:
        """Return the image at the selected working size."""
        src = self.original_img
        if src is None:
            return None, ""
        h, w = src.shape[:2]
        limit = self.work_limit
        if limit is not None and max(h, w) > int(limit):
            scale = float(limit) / float(max(h, w))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            return as_uint8(resize_bilinear(src, nw, nh)), \
                f"working at {nw}×{nh} (source {w}×{h})"
        return src.copy(), ""

    def _set_input(self, arr: np.ndarray, name: str, note: str = ""):
        self.original_img = arr
        self.source_name = name
        work, work_note = self._working_copy()
        self.work_img = work
        if work is None:
            return

        self.input_view.set_image(work)
        self.input_dims.setText(self._dims_text(work))
        self._update_status_info()
        self.setWindowTitle(f"DIP Lab — {name}")
        message = f"Loaded {name}"
        details = " · ".join(x for x in (note, work_note) if x)
        if details:
            message += f" — {details}"
        self._apply_now(force=True)
        self.statusBar().showMessage(message)

    def _on_size_changed(self, *_):
        text = self.size_combo.currentText()
        self.work_limit = None if text.lower().startswith("full") \
            else int(text.split()[0])
        if self.original_img is None:
            return
        work, note = self._working_copy()
        if work is None:
            return
        self.work_img = work
        self.input_view.set_image(work)
        self.input_dims.setText(self._dims_text(work))
        self._update_status_info()
        self._apply_now(force=True)
        self.statusBar().showMessage(
            f"Working size: {text}" + (f" — {note}" if note else ""))

    def _reset_input(self):
        if self.original_img is None:
            return
        work, _ = self._working_copy()
        if work is None:
            return
        self.work_img = work
        self.input_view.set_image(self.work_img)
        self.input_dims.setText(self._dims_text(self.work_img))
        self._update_status_info()
        self._apply_now(force=True)
        self.statusBar().showMessage("Input reset to the original image")

    def _use_result(self):
        if self.result_img is None:
            return
        self.work_img = self.result_img.copy()
        self.input_view.set_image(self.work_img)
        self.input_dims.setText(self._dims_text(self.work_img))
        self._apply_now(force=True)
        self.statusBar().showMessage(
            "Result is now the input — apply another operation to chain them")

    def _save_result(self):
        if self.result_img is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Apply an operation first.")
            return
        stem = os.path.splitext(self.source_name)[0] or "result"
        default = os.path.join(os.path.expanduser("~"), f"{stem}_result.png")
        path, selected = QFileDialog.getSaveFileName(
            self, "Save result", default,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;"
            "BMP (*.bmp)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            ext = ".jpg" if "JPEG" in selected else (
                ".tif" if "TIFF" in selected else ".bmp" if "BMP" in selected
                else ".png")
            path += ext
        try:
            Image.fromarray(self.result_img).save(path)
        except Exception as exc:              # noqa: BLE001
            QMessageBox.critical(self, "Cannot save image", str(exc))
            return
        self.statusBar().showMessage(f"Saved to {path}")

    # ------------------------------------------------------------------
    # drag & drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                self._load_path(path)
                break

    # ------------------------------------------------------------------
    # dialogs
    # ------------------------------------------------------------------
    def _text_dialog(self, title: str, text: str, size=(720, 560)):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        v = QVBoxLayout(dlg)
        edit = QPlainTextEdit(text)
        edit.setReadOnly(True)
        edit.setFont(QFont("Monospace", 9))
        edit.setMinimumSize(*size)
        v.addWidget(edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        v.addWidget(buttons)
        dlg.exec_()

    def _show_dft_demo(self):
        self._text_dialog("DFT worked example — Lecture 4",
                          fr.dft_worked_example(), (760, 600))

    def _show_about(self):
        lines = [
            "DIP Lab — every algorithm below is implemented from scratch",
            "with NumPy in the diplab/ package (no OpenCV).",
            "",
            "Notes 00 — Foundations & Enhancement (Lec 1-2)",
            "  • pixel matrices, RGB -> grayscale (average / weighted)",
            "  • negative, log, gamma, piecewise linear",
            "  • contrast stretching, histogram equalization, LUTs",
            "",
            "Notes 01 — Spatial Filters (Lec 3)",
            "  • average, weighted 1/16, median, min, max, nth-order",
            "  • adaptive local noise reduction",
            "  • Sobel / Scharr, Laplacian, Laplacian sharpening",
            "  • border padding: reflect / replicate / zeros",
            "",
            "Notes 02 — Frequency Domain (Lec 4)",
            "  • 1-D/2-D DFT and a radix-2 FFT (both from scratch)",
            "  • spectrum visualisation, ideal/Butterworth/Gaussian LP & HP",
            "  • band-pass, band-reject, notch, homomorphic filtering",
            "",
            "Notes 03 — Image Features (Lec 5)",
            "  • HOG: gradients, soft-binned cells, block normalisation",
            "  • Harris corners: structure tensor, R response, NMS",
            "",
            f"{len(OPERATIONS)} operations in total.",
            "",
            "Tips:",
            "  • scroll over an image to zoom, double-click to fit",
            "  • Ctrl+O open, Ctrl+S save, Ctrl+R reset, Ctrl+U chain result",
            "  • drop an image file anywhere on the window",
        ]
        self._text_dialog("Algorithm map", "\n".join(lines), (760, 660))
