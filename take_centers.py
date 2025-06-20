import os
import tkinter as tk
import json
from Constants import *
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageMarkerApp:
    def __init__(self, root: tk.Tk, image_folder):
        self.root = root
        self.root.title("Image Marker")
        
        # Словарь для хранения точек: {имя_файла: [(x1,y1), (x2,y2), ...]}
        self.points_dict = {}
        
        # Загрузка изображений из папки
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_image_index = 0
        
        # Создание элементов GUI
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Полосы прокрутки
        self.h_scroll = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = tk.Scrollbar(root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Привязка событий
        self.canvas.bind("<Button-1>", self.add_point)
        self.root.bind("<BackSpace>", self.remove_last_point)
        self.root.bind("<Return>", self.next_image)
        
        # Загрузка первого изображения
        self.load_image()
    
    def load_image(self):
        if self.current_image_index >= len(self.image_files):
            print("Все изображения обработаны!")
            self.root.quit()
            return
        
        # Очистка холста
        self.canvas.delete("all")
        
        # Загрузка нового изображения
        self.current_image = self.image_files[self.current_image_index]
        image_path = os.path.join(self.image_folder, self.current_image)
        self.pil_image = Image.open(image_path)
        
        # Создание изображения для tkinter
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        
        # Отображение на холсте
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, self.pil_image.width, self.pil_image.height))
        
        # Загрузка существующих точек (если есть)
        if self.current_image in self.points_dict:
            for x, y in self.points_dict[self.current_image]:
                self.draw_point(x, y)
        
        print(f"Изображение: {self.current_image} | Точки: {self.points_dict.get(self.current_image, [])}")
    
    def add_point(self, event):
        # Получение координат с учетом прокрутки
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Добавление точки в словарь
        if self.current_image not in self.points_dict:
            self.points_dict[self.current_image] = []
        self.points_dict[self.current_image].append((x, y))
        
        # Отрисовка точки
        self.draw_point(x, y)
        print(f"Добавлена точка: ({x}, {y})")
    
    def draw_point(self, x, y):
        # Рисуем красный круг (5px радиус)
        radius = 2
        self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill="red", outline="black", tags="point"
        )
    
    def remove_last_point(self, event):
        if self.current_image in self.points_dict and self.points_dict[self.current_image]:
            # Удаляем последнюю точку
            self.points_dict[self.current_image].pop()
            
            # Перерисовываем изображение с оставшимися точками
            self.load_image()
            print("Удалена последняя точка")
    
    def next_image(self, event):
        # Переход к следующему изображению
        self.current_image_index += 1
        self.load_image()

if __name__ == "__main__":
    # Укажите путь к папке с изображениями  # Замените на реальный путь
    
    root = tk.Tk()
    root.geometry("1600x1080")
    app = ImageMarkerApp(root, PATH_SAVE_CALIBRE_PROJECTOR)
    root.mainloop()
    
    # После закрытия окна выведем все собранные точки
    print("\nСобранные точки:")
    for img, points in app.points_dict.items():
        print(f"{img}: {points}")
    
    with open("centers.json", "w") as f:
        json.dump(app.points_dict, f)