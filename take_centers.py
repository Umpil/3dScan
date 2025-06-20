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
        
        self.points_dict = {}
        
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_image_index = 0
        
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.h_scroll = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = tk.Scrollbar(root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.bind("<Button-1>", self.add_point)
        self.root.bind("<BackSpace>", self.remove_last_point)
        self.root.bind("<Return>", self.next_image)
        
        self.load_image()
    
    def load_image(self):
        if self.current_image_index >= len(self.image_files):
            print("Все изображения обработаны!")
            self.root.quit()
            return
        
        # Очистка холста
        self.canvas.delete("all")
        
        self.current_image = self.image_files[self.current_image_index]
        image_path = os.path.join(self.image_folder, self.current_image)
        self.pil_image = Image.open(image_path)
        
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, self.pil_image.width, self.pil_image.height))
        
        if self.current_image in self.points_dict:
            for x, y in self.points_dict[self.current_image]:
                self.draw_point(x, y)
        
        print(f"Изображение: {self.current_image} | Точки: {self.points_dict.get(self.current_image, [])}")
    
    def add_point(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        

        if self.current_image not in self.points_dict:
            self.points_dict[self.current_image] = []
        self.points_dict[self.current_image].append((x, y))
        

        self.draw_point(x, y)
        print(f"Добавлена точка: ({x}, {y})")
    
    def draw_point(self, x, y):

        radius = 2
        self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill="red", outline="black", tags="point"
        )
    
    def remove_last_point(self, event):
        if self.current_image in self.points_dict and self.points_dict[self.current_image]:

            self.points_dict[self.current_image].pop()
            

            self.load_image()
            print("Удалена последняя точка")
    
    def next_image(self, event):

        self.current_image_index += 1
        self.load_image()

if __name__ == "__main__":
   
    root = tk.Tk()
    root.geometry("1600x1080")
    app = ImageMarkerApp(root, PATH_SAVE_CALIBRE_PROJECTOR)
    root.mainloop()
    
    print("\nСобранные точки:")
    for img, points in app.points_dict.items():
        print(f"{img}: {points}")
    
    with open("centers.json", "w") as f:
        json.dump(app.points_dict, f)