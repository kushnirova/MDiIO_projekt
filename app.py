import tkinter as tk

from gui import ImageApp


def main() -> None:
    root = tk.Tk()
    ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
