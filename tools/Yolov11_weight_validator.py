import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from ultralytics import YOLO
from threading import Thread
from PIL import Image, ImageTk
import json

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Validator-weight-test")

        # Initialize variables
        self.image_path = None  # Variable to store the image path
        self.inferencing = False
        self.model = YOLO("bestV3-OBB.pt")
        self.current_weight_path = "bestV3-OBB.pt"  # Default weight path

        # Create UI
        self.create_ui()

    def create_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, pady=5)

        # Button to select an image file
        self.select_image_button = tk.Button(top_frame, text="Select Image", command=self.select_image)
        self.select_image_button.pack(side=tk.LEFT, padx=5)

        # Canvas to show the image
        self.canvas = tk.Canvas(self.root, width=640, height=360)  # Adjusted to 16:9 aspect ratio
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)

        # Button to start/stop inference
        self.control_button = tk.Button(top_frame, text="Start Inference", command=self.toggle_inference)
        self.control_button.pack(side=tk.LEFT, padx=5)

        # Button to select weight file
        self.select_weight_button = tk.Button(top_frame, text="Select Weight", command=self.select_weight)
        self.select_weight_button.pack(side=tk.LEFT, padx=5)

        # Label to display current weight path
        self.weight_label = tk.Label(top_frame, text=f"Current Weight: {self.current_weight_path}")
        self.weight_label.pack(side=tk.LEFT, padx=5)

    def select_image(self):
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.display_image()

    def display_image(self):
        if self.image_path:
            image = cv2.imread(self.image_path)

            if self.inferencing:
                # Run YOLO inference and annotate the image
                results = self.model(image, conf=0.8)
                annotated_frame = results[0].plot()    
                self.save_inference_to_json(results)

            else:
                annotated_frame = image

            # Convert the frame to a format suitable for Tkinter
            frame_resized = cv2.resize(annotated_frame, (640, 360))  # Resize to 16:9 aspect ratio
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

    def toggle_inference(self):
        # Toggle the inference state
        self.inferencing = not self.inferencing
        self.control_button.config(text="Stop Inference" if self.inferencing else "Start Inference")
        self.display_image()  # Refresh the image to show the updated inference state

    def select_weight(self):
        # Open a file dialog to select a weight file
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pt")])
        if file_path:
            self.current_weight_path = file_path
            self.model = YOLO(file_path)  # Load the selected model file
            self.weight_label.config(text=f"Current Weight: {self.current_weight_path}")

    def save_inference_to_json(self, results, output_file="hasil.json"):
        """
        Save YOLO inference results to a JSON file in xywhr format.

        Args:
            results: The inference results from the YOLO model.
            output_file: The path to the JSON file to save the results.
        """
        inference_data = []

        inference_data.append({"image_path": self.image_path})

        for result in results:
            obb = result.obb
            for i in range(len(obb.conf)):
                class_name = self.model.names[int(obb.cls[i])]
                inference_data.append({
                    "x": float(obb.xywhr[i, 0]),  # Center X-coordinate
                    "y": float(obb.xywhr[i, 1]),  # Center Y-coordinate
                    "w": float(obb.xywhr[i, 2]),  # Width
                    "h": float(obb.xywhr[i, 3]),  # Height
                    "r": float(obb.xywhr[i, 4]),  # Rotation angle
                    "confidence": float(obb.conf[i]),  # Confidence score
                    "class": class_name  # Class index
                })

        # Save results to JSON
        with open(output_file, "w") as json_file:
            json.dump(inference_data, json_file, indent=4)
        print(f"Inference results saved to {output_file}")

    def on_close(self):
        # Stop the application
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
