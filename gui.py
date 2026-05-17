import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import cv2
import numpy as np
from PIL import Image, ImageTk

from batch_ops import batch_resize_images, batch_watermark_images
from io_utils import read_image, write_image
from processing import (
    apply_ann_processing,
    build_ann_model,
    detect_edges,
    fit_to_preview_size,
    threshold_image,
)


class ImageApp:
    def __init__(self, root) -> None:
        self.root = root
        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.current_path: str | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_scale: float = 1.0
        self.preview_offset_x: int = 0
        self.preview_offset_y: int = 0
        self.preview_display_size: tuple[int, int] = (0, 0)
        self.crop_rect: tuple[int, int, int, int] | None = None
        self.crop_drag_mode: str | None = None
        self.crop_drag_handle: str | None = None
        self.crop_drag_start: tuple[int, int] | None = None
        self.crop_drag_rect: tuple[int, int, int, int] | None = None
        self.blur_dragging: bool = False
        self.blur_last_point: tuple[int, int] | None = None
        self.blur_cursor_point: tuple[int, int] | None = None
        self.ann_model = build_ann_model()
        self.crop_ratios = {
            "Swobodne": None,
            "Kwadrat 1:1": 1.0,
            "16:9": 16 / 9,
            "3:4": 3 / 4,
            "4:3": 4 / 3,
            "9:16": 9 / 16,
        }
        self.crop_ratio = tk.StringVar(value="Swobodne")
        self.active_tab = "edge"
        self.blur_brush_size = tk.IntVar(value=45)
        self.blur_strength = tk.IntVar(value=9)
        
        # Mapowanie fontów OpenCV
        self.FONTS = {
            "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
            "Plain": cv2.FONT_HERSHEY_PLAIN,
            "Duplex": cv2.FONT_HERSHEY_DUPLEX,
            "Complex": cv2.FONT_HERSHEY_COMPLEX,
            "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
        }

        self._build_ui()

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=15)
        root_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel z kontrolkami
        left_panel = ttk.Frame(root_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        left_panel.pack_propagate(False)
        left_panel.config(width=420)

        # Prawa panel z podglądem
        right_panel = ttk.LabelFrame(root_frame, text="Podgląd")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        right_panel_inner = ttk.Frame(right_panel, padding=10)
        right_panel_inner.pack(fill=tk.BOTH, expand=True)

        # Top actions - lepiej przycisków
        top_actions = ttk.Frame(left_panel)
        top_actions.pack(fill=tk.X, pady=(0, 15))

        ttk.Button(top_actions, text="📂 Wczytaj", command=self.load_image, bootstyle="primary").pack(side=tk.LEFT, padx=3)
        ttk.Button(top_actions, text="💾 Zapisz", command=self.save_current_image, bootstyle="info").pack(side=tk.LEFT, padx=3)
        ttk.Button(top_actions, text="↺ Reset", command=self.reset_image, bootstyle="warning").pack(side=tk.LEFT, padx=3)

        # Image info
        self.image_info = tk.StringVar(value="Brak wczytanego obrazu")
        info_label = ttk.Label(left_panel, textvariable=self.image_info, wraplength=380, font=("TkDefaultFont", 9))
        info_label.pack(fill=tk.X, pady=(0, 15))

        # Tabs z kartami ułożone w dwa rzędy
        self.tab_buttons: dict[str, ttk.Button] = {}
        self.tab_frames: dict[str, ttk.Frame] = {}

        tab_header = ttk.Frame(left_panel)
        tab_header.pack(fill=tk.X, pady=(0, 10))
        tab_row1 = ttk.Frame(tab_header)
        tab_row1.pack(fill=tk.X, pady=(0, 4))
        tab_row2 = ttk.Frame(tab_header)
        tab_row2.pack(fill=tk.X, pady=(0, 4))
        tab_row3 = ttk.Frame(tab_header)
        tab_row3.pack(fill=tk.X)

        tabs_area = ttk.Frame(left_panel)
        tabs_area.pack(fill=tk.BOTH, expand=True)

        edge_tab = ttk.Frame(tabs_area, padding=12)
        threshold_tab = ttk.Frame(tabs_area, padding=12)
        batch_resize_tab = ttk.Frame(tabs_area, padding=12)
        batch_watermark_tab = ttk.Frame(tabs_area, padding=12)
        crop_tab = ttk.Frame(tabs_area, padding=12)
        ann_tab = ttk.Frame(tabs_area, padding=12)
        blur_tab = ttk.Frame(tabs_area, padding=12)

        self._register_tab("edge", edge_tab, "🔍 Krawędzie", tab_row1)
        self._register_tab("threshold", threshold_tab, "⚫ Progowanie", tab_row1)
        self._register_tab("resize", batch_resize_tab, "📐 Resize", tab_row1)
        self._register_tab("watermark", batch_watermark_tab, "💧 Watermark", tab_row2)
        self._register_tab("crop", crop_tab, "✂ Przycinanie", tab_row2)
        self._register_tab("ann", ann_tab, "🧠 Sieć NN", tab_row2)
        self._register_tab("blur", blur_tab, "🫧 Blur", tab_row3)

        self._build_edge_tab(edge_tab)
        self._build_threshold_tab(threshold_tab)
        self._build_resize_tab(batch_resize_tab)
        self._build_watermark_tab(batch_watermark_tab)
        self._build_crop_tab(crop_tab)
        self._build_ann_tab(ann_tab)
        self._build_blur_tab(blur_tab)

        self._show_tab("edge")

        # Preview canvas
        self.preview_canvas = tk.Canvas(right_panel_inner, bg="#111111", highlightthickness=0, cursor="crosshair")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", self._on_preview_canvas_configure)
        self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_mouse_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_mouse_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_preview_mouse_release)
        self.preview_canvas.bind("<Motion>", self._on_preview_mouse_move)
        self.preview_canvas.bind("<Leave>", self._on_preview_mouse_leave)
        self.preview_canvas.create_text(
            10,
            10,
            anchor=tk.NW,
            fill="#dddddd",
            text="Wczytaj obraz, aby użyć podglądu i przycinania.",
            tags=("placeholder",),
        )

    def _register_tab(self, tab_key: str, tab_frame: ttk.Frame, label: str, row: ttk.Frame) -> None:
        tab_frame.pack_forget()
        self.tab_frames[tab_key] = tab_frame

        button = ttk.Button(row, text=label, bootstyle="secondary-outline", command=lambda key=tab_key: self._show_tab(key))
        button.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
        self.tab_buttons[tab_key] = button

    def _show_tab(self, tab_key: str) -> None:
        self.active_tab = tab_key
        for key, frame in self.tab_frames.items():
            if key == tab_key:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

        for key, button in self.tab_buttons.items():
            button.configure(bootstyle="primary" if key == tab_key else "secondary-outline")

        self.crop_drag_mode = None
        self.crop_drag_handle = None
        self.crop_drag_start = None
        self.crop_drag_rect = None
        self.blur_dragging = False
        self.blur_last_point = None
        self.blur_cursor_point = None
        if hasattr(self, "preview_canvas"):
            if tab_key == "crop":
                self.preview_canvas.configure(cursor="crosshair")
            elif tab_key == "blur":
                self.preview_canvas.configure(cursor="none")
            else:
                self.preview_canvas.configure(cursor="arrow")
            self._redraw_preview()

    def _build_edge_tab(self, parent: ttk.Frame) -> None:
        title = ttk.Label(parent, text="Wykrywanie krawędzi", font=("TkDefaultFont", 11, "bold"))
        title.pack(anchor=tk.W, pady=(0, 10))
        
        desc = ttk.Label(parent, text="Wybierz algorytm do detekcji krawędzi:", font=("TkDefaultFont", 9))
        desc.pack(anchor=tk.W, pady=(0, 8))

        self.edge_choice = tk.StringVar(value="Canny")
        for option in ("Canny", "Sobel", "Laplacian"):
            ttk.Radiobutton(parent, text=option, variable=self.edge_choice, value=option).pack(anchor=tk.W, pady=4)

        ttk.Button(parent, text="▶ Wykonaj", command=self.apply_edge_detection, bootstyle="success").pack(anchor=tk.W, pady=(12, 0))

    def _build_threshold_tab(self, parent: ttk.Frame) -> None:
        title = ttk.Label(parent, text="Progowanie obrazu", font=("TkDefaultFont", 11, "bold"))
        title.pack(anchor=tk.W, pady=(0, 10))
        
        desc = ttk.Label(parent, text="Wybierz algorytm progowania:", font=("TkDefaultFont", 9))
        desc.pack(anchor=tk.W, pady=(0, 8))

        self.threshold_choice = tk.StringVar(value="Binary")
        for option, label in (
            ("Binary", "Binarne (z progiem ręcznym)"),
            ("Otsu", "Otsu"),
            ("Adaptive", "Adaptive Gaussian"),
        ):
            ttk.Radiobutton(
                parent,
                text=label,
                variable=self.threshold_choice,
                value=option,
                command=self._toggle_binary_threshold_controls,
            ).pack(anchor=tk.W, pady=4)

        self.binary_threshold_controls = ttk.Frame(parent)
        self.binary_threshold_controls.pack(fill=tk.X, pady=(12, 0))
        
        ttk.Label(self.binary_threshold_controls, text="Wartość progu (0-255):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))

        self.binary_threshold = tk.IntVar(value=127)
        self.binary_threshold_text = tk.StringVar(value="127")
        self.binary_threshold_scale = ttk.Scale(
            self.binary_threshold_controls,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.binary_threshold,
            command=self._on_binary_threshold_slide,
        )
        self.binary_threshold_scale.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(self.binary_threshold_controls, textvariable=self.binary_threshold_text, font=("TkDefaultFont", 10)).pack(anchor=tk.W)

        self._toggle_binary_threshold_controls()
        ttk.Button(parent, text="▶ Wykonaj", command=self.apply_threshold, bootstyle="success").pack(anchor=tk.W, pady=(12, 0))

    def _on_binary_threshold_slide(self, value: str) -> None:
        threshold = int(round(float(value)))
        self.binary_threshold_text.set(str(threshold))

    def _toggle_binary_threshold_controls(self) -> None:
        if self.threshold_choice.get() == "Binary":
            self.binary_threshold_controls.pack(fill=tk.X, pady=(8, 0))
        else:
            self.binary_threshold_controls.pack_forget()

    def _build_resize_tab(self, parent: ttk.Frame) -> None:
        title = ttk.Label(parent, text="Zmiana rozdzielczości", font=("TkDefaultFont", 11, "bold"))
        title.pack(anchor=tk.W, pady=(0, 10))
        
        desc = ttk.Label(parent, text="Zmieni rozmiar wszystkich obrazów w folderze:", font=("TkDefaultFont", 9))
        desc.pack(anchor=tk.W, pady=(0, 8))

        self.resize_input_dir = tk.StringVar()
        self.resize_output_dir = tk.StringVar()
        self.resize_width = tk.IntVar(value=800)
        self.resize_height = tk.IntVar(value=600)

        self._dir_picker(parent, "Folder wejściowy:", self.resize_input_dir).pack(fill=tk.X, pady=(0, 6))
        self._dir_picker(parent, "Folder wyjściowy:", self.resize_output_dir).pack(fill=tk.X, pady=(0, 12))

        ttk.Label(parent, text="Docelowy rozmiar:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 6))
        
        size_row = ttk.Frame(parent)
        size_row.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(size_row, text="Szer:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(size_row, textvariable=self.resize_width, width=8).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(size_row, text="Wys:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(size_row, textvariable=self.resize_height, width=8).pack(side=tk.LEFT)

        ttk.Button(parent, text="▶ Przetwórz folder", command=self.batch_resize, bootstyle="success").pack(anchor=tk.W)

    def _build_watermark_tab(self, parent: ttk.Frame) -> None:
        title = ttk.Label(parent, text="Dodanie znaku wodnego", font=("TkDefaultFont", 11, "bold"))
        title.pack(anchor=tk.W, pady=(0, 10))
        
        desc = ttk.Label(parent, text="Nakładanie watermarki na wszystkie obrazy w folderze:", font=("TkDefaultFont", 9))
        desc.pack(anchor=tk.W, pady=(0, 8))

        self.wm_input_dir = tk.StringVar()
        self.wm_output_dir = tk.StringVar()
        self.wm_text = tk.StringVar(value="Znak wodny")
        self.wm_opacity = tk.DoubleVar(value=0.4)
        self.wm_scale = tk.DoubleVar(value=1.0)
        self.wm_position = tk.StringVar(value="Prawy dolny")
        self.wm_font = tk.StringVar(value="Simplex")
        self.wm_color = (255, 255, 255)  # BGR format

        self._dir_picker(parent, "Folder wejściowy:", self.wm_input_dir).pack(fill=tk.X, pady=(0, 6))
        self._dir_picker(parent, "Folder wyjściowy:", self.wm_output_dir).pack(fill=tk.X, pady=(0, 12))

        ttk.Label(parent, text="Treść:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Entry(parent, textvariable=self.wm_text).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="Pozycja:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        position_combo = ttk.Combobox(
            parent,
            textvariable=self.wm_position,
            values=("Lewy górny", "Prawy górny", "Lewy dolny", "Prawy dolny", "Środek"),
            state="readonly",
        )
        position_combo.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="Font:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        
        font_frame = ttk.Frame(parent)
        font_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Mapowanie fontów tkinter na style fontów
        font_styles = {
            "Simplex": ("Courier", 10, "normal"),
            "Plain": ("Courier", 8, "normal"),
            "Duplex": ("Times", 10, "normal"),
            "Complex": ("Courier", 10, "bold"),
            "Triplex": ("Times", 11, "bold"),
        }
        
        for font_name in list(self.FONTS.keys()):
            tkfont_tuple = font_styles.get(font_name, ("TkDefaultFont", 9, "normal"))
            tk.Radiobutton(
                font_frame,
                text=font_name,
                variable=self.wm_font,
                value=font_name,
                font=tkfont_tuple,
                anchor=tk.W
            ).pack(anchor=tk.W, pady=1)

        color_frame = ttk.Frame(parent)
        color_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(color_frame, text="Kolor:", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        
        self.color_button = tk.Button(
            color_frame,
            text="   Wybierz kolor   ",
            command=self.choose_watermark_color,
            bg="#{:02x}{:02x}{:02x}".format(*reversed(self.wm_color)),
            fg="white",
            font=("TkDefaultFont", 9),
            cursor="hand2"
        )
        self.color_button.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(parent, text="Skala (0.3 - 3.0):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Scale(parent, from_=0.3, to=3.0, orient=tk.HORIZONTAL, variable=self.wm_scale).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="Przezroczystość (0.1 - 1.0):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Scale(parent, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.wm_opacity).pack(fill=tk.X, pady=(0, 12))

        ttk.Button(parent, text="▶ Przetwórz folder", command=self.batch_watermark, bootstyle="success").pack(anchor=tk.W, pady=(12, 0))

    def _build_crop_tab(self, parent: ttk.Frame) -> None:
        title = ttk.Label(parent, text="Przycinanie obrazu", font=("TkDefaultFont", 11, "bold"))
        title.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(
            parent,
            text="Przeciągaj ramkę na podglądzie. Uchwyty w narożnikach zmieniają rozmiar, a poniżej ustawisz proporcje.",
            wraplength=360,
            font=("TkDefaultFont", 9),
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(parent, text="Proporcje ramki:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ratio_frame = ttk.Frame(parent)
        ratio_frame.pack(fill=tk.X, pady=(0, 10))
        for label in self.crop_ratios:
            ttk.Radiobutton(
                ratio_frame,
                text=label,
                value=label,
                variable=self.crop_ratio,
                command=self._on_crop_ratio_change,
            ).pack(anchor=tk.W, pady=1)

        ttk.Button(parent, text="✂ Przytnij obraz", command=self.apply_crop, bootstyle="success").pack(anchor=tk.W, pady=(12, 0))

    def _build_ann_tab(self, parent: ttk.Frame) -> None:
        title = ttk.Label(parent, text="Przetwarzanie Pareidolii", font=("TkDefaultFont", 11, "bold"))
        title.pack(anchor=tk.W, pady=(0, 8))
        
        ttk.Label(
            parent,
            text=(
                "Sieć neuronowa (OpenCV ANN_MLP) wyszukuje obszary twarzopodobne "
                "i lokalnie je wzmacnia, wydobywając efekt widocznych twarzy w zwykłych obiektach."
            ),
            wraplength=360,
            font=("TkDefaultFont", 9)
        ).pack(anchor=tk.W, pady=(0, 12))

        self.ann_effect_strength = tk.DoubleVar(value=2.2)
        self.ann_volume_strength = tk.DoubleVar(value=1.9)

        ttk.Label(parent, text="Siła efektu twarzy (0.8 - 3.2):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Scale(parent, from_=0.8, to=3.2, orient=tk.HORIZONTAL, variable=self.ann_effect_strength).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="Objętość i wyginanie (0.6 - 3.0):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Scale(parent, from_=0.6, to=3.0, orient=tk.HORIZONTAL, variable=self.ann_volume_strength).pack(fill=tk.X, pady=(0, 12))

        ttk.Button(parent, text="▶ Przetwórz obraz", command=self.apply_neural_processing, bootstyle="success").pack(anchor=tk.W)

    def _build_blur_tab(self, parent: ttk.Frame) -> None:
        title = ttk.Label(parent, text="Rozmywanie pędzlem", font=("TkDefaultFont", 11, "bold"))
        title.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(
            parent,
            text=(
                "Przeciągaj myszą po podglądzie, aby lokalnie rozmywać obraz. "
                "Rozmiar pędzla i moc rozmycia możesz regulować suwakami."
            ),
            wraplength=360,
            font=("TkDefaultFont", 9),
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(parent, text="Rozmiar pędzla (5 - 200 px):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Scale(parent, from_=5, to=200, orient=tk.HORIZONTAL, variable=self.blur_brush_size).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="Moc rozmycia (1 - 40):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Scale(parent, from_=1, to=40, orient=tk.HORIZONTAL, variable=self.blur_strength).pack(fill=tk.X, pady=(0, 12))

    def _dir_picker(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=label, font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            row,
            text="📁",
            width=3,
            command=lambda: self._choose_directory(variable),
        ).pack(side=tk.LEFT, padx=(4, 0))
        return frame

    def _choose_directory(self, variable: tk.StringVar) -> None:
        directory = filedialog.askdirectory()
        if directory:
            variable.set(directory)

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[
                ("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not path:
            return

        try:
            image = read_image(path)
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Błąd", str(exc))
            return

        self.current_path = path
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.image_info.set(f"Wczytano: {path}\nRozmiar: {image.shape[1]}x{image.shape[0]}")
        self.reset_crop_frame()
        self._update_preview(self.current_image)

    def save_current_image(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        path = filedialog.asksaveasfilename(
            title="Zapisz obraz",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not path:
            return

        try:
            write_image(path, self.current_image)
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Błąd zapisu", str(exc))
            return

        messagebox.showinfo("Sukces", f"Zapisano obraz:\n{path}")

    def reset_image(self) -> None:
        if self.original_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        self.current_image = self.original_image.copy()
        self.reset_crop_frame()
        self._update_preview(self.current_image)

    def _ensure_current(self) -> np.ndarray | None:
        if self.current_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return None
        return self.current_image

    def apply_edge_detection(self) -> None:
        image = self._ensure_current()
        if image is None:
            return

        self.current_image = detect_edges(image, self.edge_choice.get())
        self.reset_crop_frame()
        self._update_preview(self.current_image)

    def apply_threshold(self) -> None:
        image = self._ensure_current()
        if image is None:
            return

        self.current_image = threshold_image(image, self.threshold_choice.get(), int(self.binary_threshold.get()))
        self.reset_crop_frame()
        self._update_preview(self.current_image)

    def batch_resize(self) -> None:
        input_dir = self.resize_input_dir.get().strip()
        output_dir = self.resize_output_dir.get().strip()
        width = int(self.resize_width.get())
        height = int(self.resize_height.get())

        if not input_dir or not output_dir:
            messagebox.showwarning("Brak folderu", "Wybierz folder wejściowy i wyjściowy.")
            return
        if width <= 0 or height <= 0:
            messagebox.showwarning("Błędny rozmiar", "Szerokość i wysokość muszą być > 0.")
            return

        processed, skipped, errors = batch_resize_images(input_dir, output_dir, width, height)
        result = f"Przetworzono: {processed}\nPominięto: {skipped}\nFolder wyjściowy: {output_dir}"
        if errors:
            result += "\n\nBłędy:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                result += f"\n... oraz {len(errors) - 5} kolejnych."
        messagebox.showinfo("Resize zakończony", result)

    def batch_watermark(self) -> None:
        input_dir = self.wm_input_dir.get().strip()
        output_dir = self.wm_output_dir.get().strip()
        text = self.wm_text.get().strip()
        opacity = float(self.wm_opacity.get())
        scale = float(self.wm_scale.get())
        position = self.wm_position.get()
        font_type = self.FONTS[self.wm_font.get()]
        color = self.wm_color

        if not input_dir or not output_dir:
            messagebox.showwarning("Brak folderu", "Wybierz folder wejściowy i wyjściowy.")
            return
        if not text:
            messagebox.showwarning("Brak tekstu", "Podaj treść znaku wodnego.")
            return

        processed, skipped, errors = batch_watermark_images(
            input_dir, output_dir, text, opacity, scale, position, color, font_type
        )
        result = f"Przetworzono: {processed}\nPominięto: {skipped}\nFolder wyjściowy: {output_dir}"
        if errors:
            result += "\n\nBłędy:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                result += f"\n... oraz {len(errors) - 5} kolejnych."
        messagebox.showinfo("Znak wodny zakończony", result)

    def apply_crop(self) -> None:
        image = self._ensure_current()
        if image is None or self.crop_rect is None:
            return

        x1, y1, x2, y2 = self.crop_rect
        if x2 <= x1 or y2 <= y1:
            messagebox.showwarning("Błędny obszar", "Wybrany obszar przycinania jest nieprawidłowy.")
            return

        self.current_image = image[y1:y2, x1:x2].copy()
        self.image_info.set(f"Przycięto obraz: {x1}, {y1}, {x2 - x1}, {y2 - y1}")
        self.reset_crop_frame()
        self._update_preview(self.current_image)

    def reset_crop_frame(self) -> None:
        image = self.current_image
        if image is None:
            self.crop_rect = None
            self._redraw_preview()
            return

        img_height, img_width = image.shape[:2]
        aspect = self._selected_crop_aspect()

        if aspect is None:
            width = max(40, int(img_width * 0.6))
            height = max(40, int(img_height * 0.6))
        else:
            width = max(40, int(img_width * 0.7))
            height = int(round(width / aspect))
            if height > int(img_height * 0.7):
                height = max(40, int(img_height * 0.7))
                width = int(round(height * aspect))

        width = min(width, img_width)
        height = min(height, img_height)
        x1 = max(0, (img_width - width) // 2)
        y1 = max(0, (img_height - height) // 2)
        x2 = min(img_width, x1 + width)
        y2 = min(img_height, y1 + height)
        self.crop_rect = (x1, y1, x2, y2)
        self._redraw_preview()

    def _selected_crop_aspect(self) -> float | None:
        return self.crop_ratios.get(self.crop_ratio.get())

    def _on_crop_ratio_change(self) -> None:
        if self.current_image is None or self.crop_rect is None:
            return

        aspect = self._selected_crop_aspect()
        if aspect is None:
            return

        img_height, img_width = self.current_image.shape[:2]
        rect = self._fit_rect_to_aspect(self.crop_rect, aspect, img_width, img_height)
        if rect is not None:
            self.crop_rect = rect
            self._redraw_preview()

    def _fit_rect_to_aspect(
        self,
        rect: tuple[int, int, int, int],
        aspect: float,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = rect
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = max(10, x2 - x1)
        height = max(10, y2 - y1)

        if width / height > aspect:
            width = int(round(height * aspect))
        else:
            height = int(round(width / aspect))

        width = min(width, img_width)
        height = min(height, img_height)
        return self._rect_from_center(center_x, center_y, width, height, img_width, img_height)

    def _on_preview_canvas_configure(self, event: tk.Event) -> None:
        self._redraw_preview()

    def _on_preview_mouse_press(self, event: tk.Event) -> None:
        if self.current_image is None:
            return

        image_point = self._canvas_to_image_point(event.x, event.y)
        if image_point is None:
            return

        if self.active_tab == "blur":
            self.blur_dragging = True
            self.blur_last_point = image_point
            self.blur_cursor_point = image_point
            self._paint_blur_at(*image_point)
            return

        if self.active_tab != "crop" or self.crop_rect is None:
            return

        handle = self._hit_test_crop_handle(event.x, event.y)
        if handle is not None:
            self.crop_drag_mode = "resize"
            self.crop_drag_handle = handle
        elif self._point_in_crop_rect(*image_point):
            self.crop_drag_mode = "move"
            self.crop_drag_handle = None
        else:
            self.crop_drag_mode = None
            self.crop_drag_handle = None
            return

        self.crop_drag_start = image_point
        self.crop_drag_rect = self.crop_rect

    def _on_preview_mouse_drag(self, event: tk.Event) -> None:
        if self.current_image is None:
            return

        image_point = self._canvas_to_image_point(event.x, event.y)
        if image_point is None:
            return

        if self.active_tab == "blur":
            if self.blur_dragging:
                self.blur_cursor_point = image_point
                start = self.blur_last_point if self.blur_last_point is not None else image_point
                self._paint_blur_line(start, image_point)
                self.blur_last_point = image_point
            return

        if self.active_tab != "crop" or self.crop_rect is None or self.crop_drag_mode is None:
            return

        img_height, img_width = self.current_image.shape[:2]
        if self.crop_drag_mode == "move" and self.crop_drag_rect is not None and self.crop_drag_start is not None:
            start_x, start_y = self.crop_drag_start
            current_x, current_y = image_point
            delta_x = int(round(current_x - start_x))
            delta_y = int(round(current_y - start_y))
            x1, y1, x2, y2 = self.crop_drag_rect
            rect = self._normalize_rect(x1 + delta_x, y1 + delta_y, x2 + delta_x, y2 + delta_y, img_width, img_height)
            if rect is not None:
                self.crop_rect = rect
                self._redraw_preview()
            return

        if self.crop_drag_mode == "resize" and self.crop_drag_rect is not None and self.crop_drag_handle is not None:
            rect = self._resize_rect_from_handle(self.crop_drag_rect, self.crop_drag_handle, image_point, img_width, img_height)
            if rect is not None:
                self.crop_rect = rect
                self._redraw_preview()

    def _on_preview_mouse_release(self, event: tk.Event) -> None:
        self.blur_dragging = False
        self.blur_last_point = None
        self.crop_drag_mode = None
        self.crop_drag_handle = None
        self.crop_drag_start = None
        self.crop_drag_rect = None

    def _on_preview_mouse_move(self, event: tk.Event) -> None:
        if self.active_tab != "blur" or self.current_image is None:
            return

        image_point = self._canvas_to_image_point(event.x, event.y)
        if image_point != self.blur_cursor_point:
            self.blur_cursor_point = image_point
            self._update_blur_brush_overlay()

    def _on_preview_mouse_leave(self, event: tk.Event) -> None:
        if self.blur_cursor_point is not None:
            self.blur_cursor_point = None
            if self.active_tab == "blur":
                self._update_blur_brush_overlay()

    def _paint_blur_line(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        x1, y1 = start
        x2, y2 = end
        distance = int(round(np.hypot(x2 - x1, y2 - y1)))
        if distance <= 0:
            self._paint_blur_at(x1, y1)
            return

        step = max(1, int(round(max(1, int(self.blur_brush_size.get())) / 4)))
        for i in range(0, distance + 1, step):
            t = i / distance
            px = int(round(x1 + (x2 - x1) * t))
            py = int(round(y1 + (y2 - y1) * t))
            self._paint_blur_at(px, py, redraw=False)
        self._paint_blur_at(x2, y2, redraw=False)
        self._redraw_preview()

    def _paint_blur_at(self, image_x: int, image_y: int, redraw: bool = True) -> None:
        if self.current_image is None:
            return

        img_height, img_width = self.current_image.shape[:2]
        radius = max(2, int(round(self.blur_brush_size.get() / 2)))
        blur_strength = max(1, int(round(self.blur_strength.get())))
        kernel_size = max(3, blur_strength * 2 + 1)

        x1 = max(0, image_x - radius)
        y1 = max(0, image_y - radius)
        x2 = min(img_width, image_x + radius + 1)
        y2 = min(img_height, image_y + radius + 1)
        if x2 <= x1 or y2 <= y1:
            return

        roi = self.current_image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.circle(mask, (image_x - x1, image_y - y1), radius, 255, -1)
        feather_sigma = max(0.5, radius * 0.35)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]

        blended = (roi.astype(np.float32) * (1.0 - alpha) + blurred_roi.astype(np.float32) * alpha).astype(np.uint8)
        self.current_image[y1:y2, x1:x2] = blended
        if redraw:
            self._redraw_preview()

    def _update_preview(self, bgr_image: np.ndarray) -> None:
        self.current_image = bgr_image
        self._redraw_preview()

    def _redraw_preview(self) -> None:
        if not hasattr(self, "preview_canvas"):
            return

        self.preview_canvas.delete("all")
        if self.current_image is None:
            self.preview_canvas.create_text(
                10,
                10,
                anchor=tk.NW,
                fill="#dddddd",
                text="Wczytaj obraz, aby użyć podglądu i przycinania.",
                tags=("placeholder",),
            )
            return

        canvas_width = max(1, self.preview_canvas.winfo_width())
        canvas_height = max(1, self.preview_canvas.winfo_height())
        img_height, img_width = self.current_image.shape[:2]
        self.preview_scale = min(canvas_width / img_width, canvas_height / img_height)
        display_width = max(1, int(round(img_width * self.preview_scale)))
        display_height = max(1, int(round(img_height * self.preview_scale)))
        self.preview_offset_x = max(0, (canvas_width - display_width) // 2)
        self.preview_offset_y = max(0, (canvas_height - display_height) // 2)
        self.preview_display_size = (display_width, display_height)

        resized = cv2.resize(self.current_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.preview_photo = ImageTk.PhotoImage(pil_img)
        self.preview_canvas.create_image(self.preview_offset_x, self.preview_offset_y, anchor=tk.NW, image=self.preview_photo)

        if self.crop_rect is not None and self.active_tab == "crop":
            self._draw_crop_overlay()
        self._update_blur_brush_overlay()

    def _update_blur_brush_overlay(self) -> None:
        if not hasattr(self, "preview_canvas"):
            return

        self.preview_canvas.delete("blur_overlay")
        if self.active_tab != "blur" or self.blur_cursor_point is None:
            return
        self._draw_blur_brush_overlay()

    def _draw_crop_overlay(self) -> None:
        if self.crop_rect is None:
            return

        x1, y1, x2, y2 = self.crop_rect
        cx1, cy1 = self._image_to_canvas_point(x1, y1)
        cx2, cy2 = self._image_to_canvas_point(x2, y2)
        self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00d1ff", width=2, tags=("crop_overlay",))
        self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#000000", width=1, dash=(6, 3), tags=("crop_overlay",))

        handle_size = 6
        for handle_x, handle_y in ((cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)):
            self.preview_canvas.create_rectangle(
                handle_x - handle_size,
                handle_y - handle_size,
                handle_x + handle_size,
                handle_y + handle_size,
                outline="#ffffff",
                fill="#00d1ff",
                tags=("crop_overlay",),
            )

    def _draw_blur_brush_overlay(self) -> None:
        if self.blur_cursor_point is None:
            return

        image_x, image_y = self.blur_cursor_point
        center_x, center_y = self._image_to_canvas_point(image_x, image_y)
        radius_px = max(2, int(round(self.blur_brush_size.get() / 2)))
        radius_canvas = max(2, int(round(radius_px * self.preview_scale)))

        self.preview_canvas.create_oval(
            center_x - radius_canvas,
            center_y - radius_canvas,
            center_x + radius_canvas,
            center_y + radius_canvas,
            outline="#000000",
            width=2,
            tags=("blur_overlay",),
        )

    def _image_to_canvas_point(self, x: int, y: int) -> tuple[int, int]:
        return (
            int(round(self.preview_offset_x + x * self.preview_scale)),
            int(round(self.preview_offset_y + y * self.preview_scale)),
        )

    def _canvas_to_image_point(self, canvas_x: int, canvas_y: int) -> tuple[int, int] | None:
        if self.current_image is None or self.preview_scale <= 0:
            return None

        img_height, img_width = self.current_image.shape[:2]
        x = int(round((canvas_x - self.preview_offset_x) / self.preview_scale))
        y = int(round((canvas_y - self.preview_offset_y) / self.preview_scale))
        if x < 0 or y < 0 or x >= img_width or y >= img_height:
            return None
        return x, y

    def _hit_test_crop_handle(self, canvas_x: int, canvas_y: int) -> str | None:
        if self.crop_rect is None:
            return None

        x1, y1, x2, y2 = self.crop_rect
        handles = {
            "nw": self._image_to_canvas_point(x1, y1),
            "ne": self._image_to_canvas_point(x2, y1),
            "sw": self._image_to_canvas_point(x1, y2),
            "se": self._image_to_canvas_point(x2, y2),
        }
        hit_size = 10
        for name, (hx, hy) in handles.items():
            if abs(canvas_x - hx) <= hit_size and abs(canvas_y - hy) <= hit_size:
                return name
        return None

    def _point_in_crop_rect(self, image_x: int, image_y: int) -> bool:
        if self.crop_rect is None:
            return False

        x1, y1, x2, y2 = self.crop_rect
        return x1 <= image_x <= x2 and y1 <= image_y <= y2

    def _resize_rect_from_handle(
        self,
        rect: tuple[int, int, int, int],
        handle: str,
        image_point: tuple[int, int],
        img_width: int,
        img_height: int,
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = rect
        mouse_x, mouse_y = image_point
        aspect = self._selected_crop_aspect()
        min_size = 20

        if handle == "nw":
            anchor_x, anchor_y = x2, y2
            target_x, target_y = mouse_x, mouse_y
            opposite = "se"
        elif handle == "ne":
            anchor_x, anchor_y = x1, y2
            target_x, target_y = mouse_x, mouse_y
            opposite = "sw"
        elif handle == "sw":
            anchor_x, anchor_y = x2, y1
            target_x, target_y = mouse_x, mouse_y
            opposite = "ne"
        else:
            anchor_x, anchor_y = x1, y1
            target_x, target_y = mouse_x, mouse_y
            opposite = "nw"

        if aspect is None:
            if handle == "nw":
                x1 = min(target_x, anchor_x - min_size)
                y1 = min(target_y, anchor_y - min_size)
                return self._normalize_rect(x1, y1, anchor_x, anchor_y, img_width, img_height)
            if handle == "ne":
                x2 = max(target_x, anchor_x + min_size)
                y1 = min(target_y, anchor_y - min_size)
                return self._normalize_rect(anchor_x, y1, x2, anchor_y, img_width, img_height)
            if handle == "sw":
                x1 = min(target_x, anchor_x - min_size)
                y2 = max(target_y, anchor_y + min_size)
                return self._normalize_rect(x1, anchor_y, anchor_x, y2, img_width, img_height)

            x2 = max(target_x, anchor_x + min_size)
            y2 = max(target_y, anchor_y + min_size)
            return self._normalize_rect(anchor_x, anchor_y, x2, y2, img_width, img_height)

        rect_by_handle = self._resize_rect_with_aspect(rect, handle, target_x, target_y, aspect, img_width, img_height)
        return rect_by_handle if rect_by_handle is not None else rect

    def _resize_rect_with_aspect(
        self,
        rect: tuple[int, int, int, int],
        handle: str,
        target_x: int,
        target_y: int,
        aspect: float,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = rect
        min_size = 20

        if handle == "nw":
            anchor_x, anchor_y = x2, y2
            dx = anchor_x - target_x
            dy = anchor_y - target_y
            if dx <= 0 or dy <= 0:
                return rect
            width, height = self._fit_aspect_to_corner(dx, dy, aspect, min_size)
            return self._normalize_rect(anchor_x - width, anchor_y - height, anchor_x, anchor_y, img_width, img_height)
        if handle == "ne":
            anchor_x, anchor_y = x1, y2
            dx = target_x - anchor_x
            dy = anchor_y - target_y
            if dx <= 0 or dy <= 0:
                return rect
            width, height = self._fit_aspect_to_corner(dx, dy, aspect, min_size)
            return self._normalize_rect(anchor_x, anchor_y - height, anchor_x + width, anchor_y, img_width, img_height)
        if handle == "sw":
            anchor_x, anchor_y = x2, y1
            dx = anchor_x - target_x
            dy = target_y - anchor_y
            if dx <= 0 or dy <= 0:
                return rect
            width, height = self._fit_aspect_to_corner(dx, dy, aspect, min_size)
            return self._normalize_rect(anchor_x - width, anchor_y, anchor_x, anchor_y + height, img_width, img_height)

        anchor_x, anchor_y = x1, y1
        dx = target_x - anchor_x
        dy = target_y - anchor_y
        if dx <= 0 or dy <= 0:
            return rect
        width, height = self._fit_aspect_to_corner(dx, dy, aspect, min_size)
        return self._normalize_rect(anchor_x, anchor_y, anchor_x + width, anchor_y + height, img_width, img_height)

    def _fit_aspect_to_corner(self, dx: int, dy: int, aspect: float, min_size: int) -> tuple[int, int]:
        width = max(min_size, dx)
        height = max(min_size, dy)
        current_aspect = width / height if height else aspect

        if current_aspect > aspect:
            width = max(min_size, int(round(height * aspect)))
        else:
            height = max(min_size, int(round(width / aspect)))

        return width, height

    def _rect_from_center(
        self,
        center_x: float,
        center_y: float,
        width: int,
        height: int,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int, int, int] | None:
        half_width = width / 2
        half_height = height / 2
        x1 = int(round(center_x - half_width))
        y1 = int(round(center_y - half_height))
        x2 = int(round(center_x + half_width))
        y2 = int(round(center_y + half_height))
        rect = self._normalize_rect(x1, y1, x2, y2, img_width, img_height)
        return rect

    def _normalize_rect(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int, int, int] | None:
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        width = right - left
        height = bottom - top
        if width < 10 or height < 10:
            return None

        if width > img_width:
            left = 0
            right = img_width
        elif left < 0:
            right -= left
            left = 0
        elif right > img_width:
            left -= right - img_width
            right = img_width

        if height > img_height:
            top = 0
            bottom = img_height
        elif top < 0:
            bottom -= top
            top = 0
        elif bottom > img_height:
            top -= bottom - img_height
            bottom = img_height

        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        if right - left < 10 or bottom - top < 10:
            return None

        return left, top, right, bottom

    def choose_watermark_color(self) -> None:
        color_rgb = colorchooser.askcolor(
            color="#{:02x}{:02x}{:02x}".format(*reversed(self.wm_color)),
            title="Wybierz kolor znaku wodnego",
        )
        if not color_rgb[1]:
            return

        rgb = color_rgb[0]
        if rgb is None:
            return

        self.wm_color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        hex_color = "#{:02x}{:02x}{:02x}".format(*reversed(self.wm_color))
        self.color_button.config(bg=hex_color)

    def apply_neural_processing(self) -> None:
        image = self._ensure_current()
        if image is None:
            return

        self.current_image = apply_ann_processing(
            image,
            self.ann_model,
            effect_strength=float(self.ann_effect_strength.get()),
            volume_strength=float(self.ann_volume_strength.get()),
        )
        self.reset_crop_frame()
        self._update_preview(self.current_image)
