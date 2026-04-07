from __future__ import annotations

import sys
from pathlib import Path

import requests

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from book_formatter import BookDocument, BookPage


class BookPageWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("bookPage")
        self._original_pixmap: QPixmap | None = None

        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.title_label.setObjectName("pageTitle")

        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setObjectName("imageView")
        self.image_view.setVisible(False)

        self.image_label = QLabel()
        self.image_label.setWordWrap(True)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("imageLabel")

        self.body_view = QTextEdit()
        self.body_view.setReadOnly(True)
        self.body_view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.body_view.setObjectName("pageBody")

        self.footer_label = QLabel()
        self.footer_label.setWordWrap(True)
        self.footer_label.setObjectName("pageFooter")

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_view)
        layout.addWidget(self.image_label)
        layout.addWidget(self.body_view, 1)
        layout.addWidget(self.footer_label)
        self.setLayout(layout)

    def render_page(self, page: BookPage, total_pages: int) -> None:
        self.title_label.setText(page.title or "")
        self.body_view.setPlainText(page.body)

        pixmap = self._load_pixmap(page)
        self._original_pixmap = pixmap
        if pixmap and not pixmap.isNull():
            self.image_view.setVisible(True)
            self._update_scaled_pixmap()
        else:
            self.image_view.clear()
            self.image_view.setVisible(False)

        image_text = ""
        if not pixmap:
            if page.image_local_path:
                image_text = f"Image: {page.image_local_path}"
            elif page.image_url:
                image_text = f"Image: {page.image_url}"
        if page.image_caption:
            image_text = f"{image_text}\n{page.image_caption}".strip()
        self.image_label.setText(image_text)
        self.image_label.setVisible(bool(image_text))

        footer_parts = [part for part in [page.footer, f"Page {page.page_number}/{total_pages}"] if part]
        self.footer_label.setText(" | ".join(footer_parts))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if not self._original_pixmap or self._original_pixmap.isNull():
            return
        target_width = max(240, self.width() - 120)
        scaled = self._original_pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
        self.image_view.setPixmap(scaled)

    def _load_pixmap(self, page: BookPage) -> QPixmap | None:
        if page.image_local_path:
            local_path = Path(page.image_local_path)
            if local_path.exists():
                pixmap = QPixmap(str(local_path))
                if not pixmap.isNull():
                    return pixmap

        if page.image_url:
            try:
                response = requests.get(
                    page.image_url,
                    timeout=10,
                    allow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0 HoverAI/1.0"},
                )
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if not content_type.startswith("image/"):
                    return None
                pixmap = QPixmap()
                pixmap.loadFromData(response.content)
                if not pixmap.isNull():
                    return pixmap
            except Exception:
                return None
        return None


class BookViewerWindow(QMainWindow):
    def __init__(self, document: BookDocument) -> None:
        super().__init__()
        self.document = document
        self.current_index = 0

        self.setWindowTitle(document.document_title or "HoverAI Book Viewer")
        self.resize(980, 720)

        self.page_widget = BookPageWidget()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.page_indicator = QLabel()
        self.page_indicator.setAlignment(Qt.AlignCenter)

        self.prev_button.clicked.connect(self.previous_page)
        self.next_button.clicked.connect(self.next_page)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.page_indicator, 1)
        nav_layout.addWidget(self.next_button)

        container = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.page_widget, 1)
        main_layout.addLayout(nav_layout)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self._install_actions()
        self._apply_styles()
        self._render_current_page()

    def _install_actions(self) -> None:
        next_action = QAction(self)
        next_action.setShortcut(QKeySequence(Qt.Key_Right))
        next_action.triggered.connect(self.next_page)
        self.addAction(next_action)

        prev_action = QAction(self)
        prev_action.setShortcut(QKeySequence(Qt.Key_Left))
        prev_action.triggered.connect(self.previous_page)
        self.addAction(prev_action)

        fullscreen_action = QAction(self)
        fullscreen_action.setShortcut(QKeySequence(Qt.Key_F11))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        self.addAction(fullscreen_action)

        exit_fullscreen_action = QAction(self)
        exit_fullscreen_action.setShortcut(QKeySequence(Qt.Key_Escape))
        exit_fullscreen_action.triggered.connect(self.exit_fullscreen)
        self.addAction(exit_fullscreen_action)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #232629;
            }
            QWidget#bookPage {
                background: #2d3136;
                border: 2px solid #50565f;
                border-radius: 16px;
            }
            QLabel#pageTitle {
                color: #ece7dc;
                font-size: 28px;
                font-weight: 700;
                padding: 12px 8px 4px 8px;
            }
            QLabel#imageView {
                background: #262b30;
                border-radius: 10px;
                margin: 8px;
                padding: 8px;
            }
            QLabel#imageLabel {
                color: #d2c7b5;
                font-size: 15px;
                padding: 8px;
                background: #3a4047;
                border-radius: 10px;
                margin: 8px;
            }
            QTextEdit#pageBody {
                color: #f3efe6;
                background: transparent;
                border: none;
                font-size: 20px;
                padding: 12px;
            }
            QLabel#pageFooter {
                color: #b8ae9f;
                font-size: 13px;
                padding: 6px 10px 10px 10px;
            }
            QPushButton {
                background: #5c6672;
                color: #f6f1e7;
                border: none;
                border-radius: 10px;
                min-height: 40px;
                padding: 0 16px;
            }
            QPushButton:disabled {
                background: #434a52;
                color: #9ea5ae;
            }
            """
        )

    def _render_current_page(self) -> None:
        total_pages = len(self.document.pages)
        if total_pages == 0:
            self.page_indicator.setText("No pages")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        page = self.document.pages[self.current_index]
        self.page_widget.render_page(page, total_pages)
        self.page_indicator.setText(f"{page.page_number} / {total_pages}")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < total_pages - 1)

    def next_page(self) -> None:
        if self.current_index < len(self.document.pages) - 1:
            self.current_index += 1
            self._render_current_page()

    def previous_page(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._render_current_page()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()


def launch_viewer(document: BookDocument) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = BookViewerWindow(document)
    window.show()
    return app.exec()
