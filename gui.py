import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OpenCV - Projekt")
        self.root.geometry("1300x800")

        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.current_path: str | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.ann_model = build_ann_model()

        self._build_ui()

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(root_frame, width=420)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(root_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        top_actions = ttk.Frame(left_panel)
        top_actions.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(top_actions, text="Wczytaj obraz", command=self.load_image).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top_actions, text="Zapisz bieżący", command=self.save_current_image).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top_actions, text="Przywróć oryginał", command=self.reset_image).pack(side=tk.LEFT)

        self.image_info = tk.StringVar(value="Brak wczytanego obrazu")
        ttk.Label(left_panel, textvariable=self.image_info, wraplength=380).pack(fill=tk.X, pady=(0, 10))

        notebook = ttk.Notebook(left_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        edge_tab = ttk.Frame(notebook, padding=10)
        threshold_tab = ttk.Frame(notebook, padding=10)
        batch_resize_tab = ttk.Frame(notebook, padding=10)
        batch_watermark_tab = ttk.Frame(notebook, padding=10)
        ann_tab = ttk.Frame(notebook, padding=10)

        notebook.add(edge_tab, text="Krawędzie")
        notebook.add(threshold_tab, text="Progowanie")
        notebook.add(batch_resize_tab, text="Resize folderu")
        notebook.add(batch_watermark_tab, text="Znak wodny")
        notebook.add(ann_tab, text="Sieć neuronowa")

        self._build_edge_tab(edge_tab)
        self._build_threshold_tab(threshold_tab)
        self._build_resize_tab(batch_resize_tab)
        self._build_watermark_tab(batch_watermark_tab)
        self._build_ann_tab(ann_tab)

        self.preview_label = ttk.Label(right_panel, text="Podgląd obrazu pojawi się tutaj", anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def _build_edge_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Wykrywanie krawędzi (3 algorytmy)").pack(anchor=tk.W)

        self.edge_choice = tk.StringVar(value="Canny")
        for option in ("Canny", "Sobel", "Laplacian"):
            ttk.Radiobutton(parent, text=option, variable=self.edge_choice, value=option).pack(anchor=tk.W, pady=2)

        ttk.Button(parent, text="Wykonaj", command=self.apply_edge_detection).pack(anchor=tk.W, pady=(8, 0))

    def _build_threshold_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Progowanie obrazu (3 algorytmy)").pack(anchor=tk.W)

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
            ).pack(anchor=tk.W, pady=2)

        self.binary_threshold_controls = ttk.Frame(parent)
        self.binary_threshold_controls.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(self.binary_threshold_controls, text="Próg dla Binarne (0-255):").pack(anchor=tk.W, pady=(0, 2))

        self.binary_threshold = tk.IntVar(value=127)
        self.binary_threshold_text = tk.StringVar(value="127")
        self.binary_threshold_scale = tk.Scale(
            self.binary_threshold_controls,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.binary_threshold,
            showvalue=False,
            command=self._on_binary_threshold_slide,
        )
        self.binary_threshold_scale.pack(fill=tk.X)
        ttk.Label(self.binary_threshold_controls, textvariable=self.binary_threshold_text).pack(anchor=tk.W)

        self._toggle_binary_threshold_controls()
        ttk.Button(parent, text="Wykonaj", command=self.apply_threshold).pack(anchor=tk.W, pady=(8, 0))

    def _on_binary_threshold_slide(self, value: str) -> None:
        threshold = int(round(float(value)))
        self.binary_threshold_text.set(str(threshold))

    def _toggle_binary_threshold_controls(self) -> None:
        if self.threshold_choice.get() == "Binary":
            self.binary_threshold_controls.pack(fill=tk.X, pady=(8, 0))
        else:
            self.binary_threshold_controls.pack_forget()

    def _build_resize_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Zmiana rozdzielczości wszystkich obrazów w folderze").pack(anchor=tk.W)

        self.resize_input_dir = tk.StringVar()
        self.resize_output_dir = tk.StringVar()
        self.resize_width = tk.IntVar(value=800)
        self.resize_height = tk.IntVar(value=600)

        self._dir_picker(parent, "Folder wejściowy:", self.resize_input_dir).pack(fill=tk.X, pady=(8, 4))
        self._dir_picker(parent, "Folder wyjściowy:", self.resize_output_dir).pack(fill=tk.X, pady=(0, 8))

        size_row = ttk.Frame(parent)
        size_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(size_row, text="Szerokość:").pack(side=tk.LEFT)
        ttk.Entry(size_row, textvariable=self.resize_width, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(size_row, text="Wysokość:").pack(side=tk.LEFT)
        ttk.Entry(size_row, textvariable=self.resize_height, width=8).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Button(parent, text="Przetwórz folder", command=self.batch_resize).pack(anchor=tk.W)

    def _build_watermark_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Dodanie znaku wodnego do wszystkich obrazów w folderze").pack(anchor=tk.W)

        self.wm_input_dir = tk.StringVar()
        self.wm_output_dir = tk.StringVar()
        self.wm_text = tk.StringVar(value="Znak wodny")
        self.wm_opacity = tk.DoubleVar(value=0.4)
        self.wm_scale = tk.DoubleVar(value=1.0)
        self.wm_position = tk.StringVar(value="Prawy dolny")

        self._dir_picker(parent, "Folder wejściowy:", self.wm_input_dir).pack(fill=tk.X, pady=(8, 4))
        self._dir_picker(parent, "Folder wyjściowy:", self.wm_output_dir).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="Treść znaku wodnego:").pack(anchor=tk.W)
        ttk.Entry(parent, textvariable=self.wm_text).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="Pozycja:").pack(anchor=tk.W)
        position_combo = ttk.Combobox(
            parent,
            textvariable=self.wm_position,
            values=("Lewy górny", "Prawy górny", "Lewy dolny", "Prawy dolny", "Środek"),
            state="readonly",
        )
        position_combo.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="Skala tekstu:").pack(anchor=tk.W)
        ttk.Scale(parent, from_=0.3, to=3.0, orient=tk.HORIZONTAL, variable=self.wm_scale).pack(fill=tk.X, pady=(0, 6))

        ttk.Label(parent, text="Przezroczystość:").pack(anchor=tk.W)
        ttk.Scale(parent, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.wm_opacity).pack(fill=tk.X, pady=(0, 8))

        ttk.Button(parent, text="Przetwórz folder", command=self.batch_watermark).pack(anchor=tk.W)

    def _build_ann_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Przetwarzanie pareidolii przez sieć neuronową (OpenCV ANN_MLP)").pack(anchor=tk.W)
        ttk.Label(
            parent,
            text=(
                "Model wyszukuje obszary twarzopodobne i lokalnie wzmacnia je, "
                "aby wydobyć efekt widocznych twarzy w zwykłych obiektach."
            ),
            wraplength=360,
        ).pack(anchor=tk.W, pady=(6, 10))

        self.ann_effect_strength = tk.DoubleVar(value=2.2)
        self.ann_volume_strength = tk.DoubleVar(value=1.9)

        ttk.Label(parent, text="Siła efektu twarzy:").pack(anchor=tk.W)
        ttk.Scale(parent, from_=0.8, to=3.2, orient=tk.HORIZONTAL, variable=self.ann_effect_strength).pack(fill=tk.X)

        ttk.Label(parent, text="Objętość i wyginanie:").pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(parent, from_=0.6, to=3.0, orient=tk.HORIZONTAL, variable=self.ann_volume_strength).pack(fill=tk.X)

        ttk.Button(parent, text="Wykonaj przetwarzanie pareidolii", command=self.apply_neural_processing).pack(
            anchor=tk.W, pady=(10, 0)
        )

    def _dir_picker(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(2, 0))
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            row,
            text="...",
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
        self._update_preview(self.current_image)

    def apply_threshold(self) -> None:
        image = self._ensure_current()
        if image is None:
            return

        self.current_image = threshold_image(image, self.threshold_choice.get(), int(self.binary_threshold.get()))
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

        if not input_dir or not output_dir:
            messagebox.showwarning("Brak folderu", "Wybierz folder wejściowy i wyjściowy.")
            return
        if not text:
            messagebox.showwarning("Brak tekstu", "Podaj treść znaku wodnego.")
            return

        processed, skipped, errors = batch_watermark_images(input_dir, output_dir, text, opacity, scale, position)
        result = f"Przetworzono: {processed}\nPominięto: {skipped}\nFolder wyjściowy: {output_dir}"
        if errors:
            result += "\n\nBłędy:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                result += f"\n... oraz {len(errors) - 5} kolejnych."
        messagebox.showinfo("Znak wodny zakończony", result)

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
        self._update_preview(self.current_image)

    def _update_preview(self, bgr_image: np.ndarray) -> None:
        preview = fit_to_preview_size(bgr_image, max_w=840, max_h=760)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.preview_photo = ImageTk.PhotoImage(pil_img)
        self.preview_label.configure(image=self.preview_photo, text="")
