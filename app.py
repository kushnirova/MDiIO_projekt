import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from gui import ImageApp


def main() -> None:
    root = ttk.Window(themename="superhero")
    root.title("OpenCV - Projekt")
    root.geometry("1400x850")
    ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
