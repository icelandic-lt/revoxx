"""User Guide Dialog for displaying the guide at startup."""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import re
import markdown2
from tkinterweb import HtmlFrame

from .dialog_utils import setup_dialog_window


class UserGuideDialog:
    """Dialog for displaying the user guide with option to show at startup."""

    def __init__(self, parent: tk.Tk, settings_manager):
        """Initialize the user guide dialog.

        Args:
            parent: Parent window
            settings_manager: Settings manager instance for saving preferences
        """
        self.parent = parent
        self.settings_manager = settings_manager

        self.dialog = tk.Toplevel(parent)
        self.dialog.withdraw()  # Hide until positioned

        self.show_at_startup = tk.BooleanVar(
            value=settings_manager.get_setting("show_user_guide_at_startup", True)
        )

        self._create_widgets()
        setup_dialog_window(
            self.dialog,
            self.parent,
            title="Revoxx User Guide",
            width=1200,
            height=800,
            center_on_parent=True,
        )

        self._load_user_guide()
        self.dialog.bind("<Escape>", lambda e: self._on_close())
        self.dialog.bind("<Return>", lambda e: self._on_close())
        self.dialog.deiconify()

    def _create_widgets(self):
        """Create and layout dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Checkbox for "show at startup"
        self.checkbox = ttk.Checkbutton(
            top_frame,
            text="Show on application startup",
            variable=self.show_at_startup,
            command=self._on_checkbox_changed,
        )
        self.checkbox.pack(side=tk.LEFT)

        self.text_widget = HtmlFrame(main_frame, messages_enabled=False)
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        close_btn = ttk.Button(
            bottom_frame, text="Close", command=self._on_close, default=tk.ACTIVE
        )
        close_btn.pack(side=tk.RIGHT)

    def _load_user_guide(self):
        """Load and display the user guide content."""
        # User guide should be in package doc folder
        guide_path = Path(__file__).parent.parent.parent / "doc" / "USER_GUIDE.md"

        if not guide_path.exists():
            error_msg = f"User Guide not found at: {guide_path}"
            error_html = f"<html><body><p style='color: red;'>{error_msg}</p></body></html>"
            self.text_widget.load_html(error_html)
            raise FileNotFoundError(error_msg)

        try:
            # Read user guide content & convert markdown to HTML
            with open(guide_path, "r", encoding="utf-8") as f:
                content = f.read()

            html_body = markdown2.markdown(
                content, extras=["tables", "fenced-code-blocks", "code-friendly"]
            )

            html_body = self._remove_links(html_body)
            html_content = self._wrap_html_with_css(html_body)
            self.text_widget.load_html(html_content)

        except Exception as e:
            error_msg = f"Error loading guide: {e}"
            error_html = f"<html><body><p>{error_msg}</p></body></html>"
            self.text_widget.load_html(error_html)

    @staticmethod
    def _remove_links(html: str) -> str:
        """Remove links and just show the text.
        We cannot open links in the standard browser and in-line rendering is actually pretty terrible.

        Args:
            html: HTML content

        Returns:
            HTML with links replaced by plain text
        """
        # Replace links with just their text content
        html = re.sub(r'<a href="[^"]+">([^<]+)</a>', r"\1", html)
        return html

    @staticmethod
    def _wrap_html_with_css(html_body: str) -> str:
        """Wrap HTML body with CSS styling for markdown2 output.

        Args:
            html_body: The HTML body content from markdown2

        Returns:
            Complete HTML with CSS
        """
        css = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 16px;
            line-height: 1.8;
            padding: 30px;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            font-size: 32px;
            margin-top: 20px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            font-size: 24px;
        }
        h3 {
            color: #7f8c8d;
            font-size: 20px;
            margin-top: 15px;
        }
        code {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            background-color: #f0f0f0;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            font-size: 14px;
            line-height: 1.4;
            overflow-x: auto;
        }
        pre code {
            background-color: transparent;
            padding: 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 15px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        ul, ol {
            padding-left: 30px;
            margin: 15px 0;
        }
        li {
            margin: 8px 0;
            line-height: 1.6;
        }
        li > ul, li > ol {
            margin: 5px 0;
        }
        p {
            margin: 15px 0;
        }
        strong {
            color: #2c3e50;
            font-weight: 600;
        }
        a {
            color: #3498db;
            text-decoration: underline;
        }
        a:hover {
            color: #2980b9;
        }
        </style>
        </head>
        <body>
        """

        footer = """
        </body>
        </html>
        """

        return css + html_body + footer

    def _on_checkbox_changed(self):
        """Handle checkbox state change."""
        # Save to preferences
        self.settings_manager.update_setting(
            "show_user_guide_at_startup", self.show_at_startup.get()
        )

    def _on_close(self):
        """Handle close button."""
        self.dialog.destroy()

    def show(self):
        """Show the dialog."""
        self.dialog.wait_window()
