import tkinter as tk
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
from pathlib import Path
import torch


class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #28+1*2-3+1 = 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), #14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), #14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), #7x7
            nn.Dropout2d(0.25) #prevents overfitting
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.convolutional_block(x)
        x = self.classifier(x)
        return x

model = CNN_MNIST()


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "MNIST_CNNmodel_weights.pth"
try:
    weights = torch.load(WEIGHTS_PATH, weights_only=True)
    model.load_state_dict(weights)
    print("Model weights loaded successfully")
except FileNotFoundError:
    print(f"Error: '{WEIGHTS_PATH.name}' not found at {WEIGHTS_PATH}")


model.eval()


class MNISTDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Predictor")
        
        self.last_x, self.last_y = None, None
        
        # visible canvas
        self.canvas = tk.Canvas(root, width=400, height=400, bg="black")
        self.canvas.pack(pady=10)
        
        # pillow image for processing
        self.pil_image = Image.new("L", (400, 400), "black")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict, font=("Arial", 14), bg="#4CAF50", fg="white")
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_canvas, font=("Arial", 14), bg="#f44336", fg="white")
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        self.result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Arial", 18, "bold"))
        self.result_label.pack(pady=15)

    def start_drawing(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.draw(event)

    def draw(self, event):
        brush_size = 30
        if self.last_x is not None and self.last_y is not None:
            # draw on tkinter
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=brush_size, fill="white", capstyle=tk.ROUND, smooth=True
            )
            # copy to pillow image
            self.pil_draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255, width=brush_size
            )
        self.last_x, self.last_y = event.x, event.y

    def stop_drawing(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="Draw a digit and click Predict!")
        # reset pillow image
        self.pil_image = Image.new("L", (400, 400), "black")
        self.pil_draw = ImageDraw.Draw(self.pil_image)

    def predict(self):
        # mnist size
        img_28x28 = self.pil_image.resize((28, 28), Image.Resampling.LANCZOS)
        
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tensor_img = transformer(img_28x28).unsqueeze(0) # Add batch dimension [1, 1, 28, 28]
        
        # model inference
        with torch.no_grad():
            output = model(tensor_img)
            probabilities = torch.softmax(output, dim=1)
            prediction = probabilities.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
        self.result_label.config(text=f"Prediction: {prediction} ({confidence:.2f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTDrawer(root)
    root.mainloop()