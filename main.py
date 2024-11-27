import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class FormDetector:
    def load_image(self, path_image):
        image = cv2.imread(path_image)
        if image is None:
            raise ValueError(f"Imagem não encontrada: {path_image}")
        return image

    def convert_to_grayscale(self, image):
        """ Transforma imagens coloridas em escala de cinza """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, image):
        """ Aplica um desfoque para reduzir os ruídos """
        return cv2.GaussianBlur(image, (5, 5), 0)

    def edge_detection(self, image):
        """ Detecta as bordas da imagem usando o método Canny """
        return cv2.Canny(image, 50, 150)

    def find_lines(self, imagem_bordas):
        """ Encontra os contornos da imagem """
        contornos, _ = cv2.findContours(
            imagem_bordas, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contornos

    def Classify_forms(self, contornos, imagem_original):
        """ Classifica as formas geométricas. Classificaremos em 5 tipos: triângulos, quadrados, círculos, pentágonos e outros """

        formas = {
            'triangulos': 0,
            'quadrados': 0,
            'circulos': 0,
            'pentagonos': 0,
            'outros': 0
        }

        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            aproximacao = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)

            # Lógica de classificação
            if len(aproximacao) == 3:  # Triângulo
                formas['triangulos'] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)

            elif len(aproximacao) == 4:  # Quadrado ou retângulo
                formas['quadrados'] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (255, 0, 0), 2)

            elif len(aproximacao) == 5:  # Pentágono
                formas['pentagonos'] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (255, 255, 0), 2)

            elif len(aproximacao) > 5:  # Círculo
                area = cv2.contourArea(contorno)
                perimetro_quadrado = perimetro * perimetro  # Corrigido para o valor correto de circularidade
                circularity = 4 * np.pi * area / perimetro_quadrado

                if circularity > 0.8:  # Se circularidade > 0.8, é um círculo
                    formas['circulos'] += 1
                    cv2.drawContours(imagem_original, [contorno], 0, (0, 0, 255), 2)

            else:
                formas['outros'] += 1

        return formas

    def view_results(self, imagem_original, formas):
        """ Visualizar os resultados e adicionar o texto com o número de triângulos, círculos e quadrados no canto inferior direito """

        # Texto com a contagem de triângulos, círculos e quadrados
        cv2.putText(imagem_original, f'Triangulos: {formas["triangulos"]}',
                    (imagem_original.shape[1] - 250, imagem_original.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(imagem_original, f'Circulos: {formas["circulos"]}',
                    (imagem_original.shape[1] - 250, imagem_original.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(imagem_original, f'Quadrados: {formas["quadrados"]}',
                    (imagem_original.shape[1] - 250, imagem_original.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(imagem_original, f'Pentagonos: {formas["pentagonos"]}',
                    (imagem_original.shape[1] - 250, imagem_original.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(imagem_original, f'Outros: {formas["outros"]}',
                    (imagem_original.shape[1] - 250, imagem_original.shape[0] - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Exibe a imagem com o contorno das formas
        img_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
        return img_rgb


# Função principal para carregar a interface gráfica com o Tkinter
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detecção de Formas Geométricas")
        self.geometry("800x600")
        self.detector = FormDetector()

        # Botão para carregar a imagem
        self.button_load = tk.Button(self, text="Carregar Imagem", command=self.load_image)
        self.button_load.pack(pady=20)

        # Canvas para exibir a imagem carregada
        self.canvas = tk.Canvas(self, width=600, height=400)
        self.canvas.pack()

        # Label para mostrar as contagens das formas
        self.result_label = tk.Label(self, text="Resultados: ", font=("Arial", 12))
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Escolha uma imagem", filetypes=[("Imagem", "*.jpg;*.png")])

        if file_path:
            imagem = self.detector.load_image(file_path)
            imagem_cinza = self.detector.convert_to_grayscale(imagem)
            imagem_blur = self.detector.apply_blur(imagem_cinza)
            imagem_bordas = self.detector.edge_detection(imagem_blur)
            contornos = self.detector.find_lines(imagem_bordas)
            formas = self.detector.Classify_forms(contornos, imagem)
            img_rgb = self.detector.view_results(imagem, formas)

            # Exibindo as contagens no label
            result_text = (f"Triângulos: {formas['triangulos']}\n"
                           f"Círculos: {formas['circulos']}\n"
                           f"Quadrados: {formas['quadrados']}\n"
                           f"Pentágonos: {formas['pentagonos']}\n"
                           f"Outros: {formas['outros']}")
            self.result_label.config(text=f"Resultados:\n{result_text}")

            # Converte para imagem que o Tkinter pode exibir
            img = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img)

            # Atualiza o Canvas com a nova imagem
            self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
            self.canvas.image = img_tk  # Salva a referência para a imagem


# Inicia a aplicação
if __name__ == "__main__":
    app = Application()
    app.mainloop()
