import tkinter as tk
from tkinter import filedialog
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class WordCloudGenerator:
    def __init__(self, master):
        self.master = master
        self.master.title("Word Cloud Generator")

        self.text_area = tk.Text(self.master, height=35, width=90)
        self.text_area.pack(pady=10)

        self.generate_button = tk.Button(self.master, text="Generate Word Cloud", command=self.generate_word_cloud)
        self.generate_button.pack(pady=5)

        self.save_button = tk.Button(self.master, text="Save Word Cloud", command=self.save_word_cloud)
        self.save_button.pack(pady=5)

        self.load_button = tk.Button(self.master, text="Load Text from File", command=self.load_text_from_file)
        self.load_button.pack(pady=5)

    def generate_word_cloud(self):
        text = self.text_area.get("1.0", tk.END)
        self.display_word_cloud(text)

    def display_word_cloud(self, text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def save_word_cloud(self):
        text = self.text_area.get("1.0", tk.END)
        self.display_word_cloud(text)

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG files", "*.png"), ("All files", "*.*")])

        if file_path:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            wordcloud.to_file(file_path)

    def load_text_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            encoding = 'utf-8'
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                    self.text_area.delete("1.0", tk.END)
                    self.text_area.insert(tk.END, text)
            except UnicodeDecodeError as e:
                print(f"Error decoding file: {e}")
def main():
    root = tk.Tk()
    app = WordCloudGenerator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
