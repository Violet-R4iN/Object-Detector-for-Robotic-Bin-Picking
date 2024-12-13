import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText  
import os
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import json

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detector for Robotic Bin-Picking")

        # Initialize variables
        self.image_path = None  # Variable to store the image path
        self.inferencing = False
        self.model = YOLO("weights\bestV3-OBB.pt")
        self.current_weight_path = "bestV3-OBB.pt"  # Default weight path
        self.selected_camera = None  # Variable to store selected camera
        self.cap = None  # VideoCapture object

        # Create UI
        self.create_ui()

    def create_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, pady=5, fill=tk.X,expand=True)

        # Button to select an image file
        self.select_image_button = tk.Button(top_frame, text="Import image", command=self.select_image)
        self.select_image_button.pack(side=tk.LEFT, padx=5)

        # Combobox to select a camera
        self.camera_combobox = ttk.Combobox(top_frame, values=self.get_available_cameras())
        self.camera_combobox.pack(side=tk.LEFT, padx=5)
        self.camera_combobox.bind("<<ComboboxSelected>>", self.on_camera_select)

        # Button to take a picture from selected camera
        self.capture_button = tk.Button(top_frame, text="Grab Camera", command=self.capture_image)
        self.capture_button.pack(side=tk.LEFT, padx=5)

        # Canvas to show the image
        self.canvas = tk.Canvas(self.root, width=640, height=360)  # Adjusted to 16:9 aspect ratio
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)

        # Add a frame for the right-side JSON preview box
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Scrollable text widget to display JSON output
        self.json_preview = ScrolledText(right_frame, wrap=tk.WORD, width=40, height=22)
        self.json_preview.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Button to start/stop inference
        self.control_button = tk.Button(top_frame, text="Start Detection", command=self.toggle_inference)
        self.control_button.pack(side=tk.LEFT, padx=5)

        # Button to select weight file
        self.select_weight_button = tk.Button(top_frame, text="Browse Weight", command=self.select_weight)
        self.select_weight_button.pack(side=tk.LEFT, padx=5)

        # Label to display current weight path
        self.weight_label = tk.Label(top_frame, text=f"Current Weight: {self.current_weight_path}")
        self.weight_label.pack(side=tk.LEFT, padx=5)

    def get_available_cameras(self, num_camera = 5):
        # Check for available cameras by attempting to open them
        cameras = []
        for i in range(num_camera):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  
            if cap.isOpened():
                # # Get the width and height of the camera
                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Append camera info to the list
                cameras.append(f"Camera {i}")
                print(f"Camera {i}")
                # print(f"Camera {i}: {width}x{height}")  
                
                cap.release()
        return cameras


    def on_camera_select(self, event):
        # Set selected camera based on Combobox selection
        selected_index = self.camera_combobox.current()
        if selected_index != -1:
            # Release the previously selected camera (if any)
            self.release_camera()

            # Extract camera port from the selected camera
            self.selected_camera = self.camera_combobox.get().split()[-1]
            print(f"Selected Camera: {self.selected_camera}")

            # Open the selected camera
            self.cap = cv2.VideoCapture(int(self.selected_camera), cv2.CAP_DSHOW)

            # Optionally check if the camera opened successfully
            if not self.cap.isOpened():
                print(f"Error: Unable to open camera {self.selected_camera}")
            else:
                print(f"Camera {self.selected_camera} opened successfully")

    def capture_image(self, width=1280, height=720):
        if self.cap:
            # Set the resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            ret, frame = self.cap.read()
            if ret:
                self.image_path = "captured_image.jpg"
                print('Image saved with resolution:', width, 'x', height)
                cv2.imwrite(self.image_path, frame)  # Save the captured image
                self.display_image()  # Display the captured image on the canvas
            else:
                print("Failed to capture image")
        else:
            print("No camera is currently open")


    def select_image(self):
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # If the camera is active, stop it before displaying the selected image
            if self.cap and self.cap.isOpened():
                self.release_camera()
                self.camera_combobox.set('')  # Clear the camera combobox

            self.image_path = file_path
            self.display_image()  # Display the selected image on the canvas


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

            # Resize the image directly to fit the canvas size 
            resized_image = cv2.resize(annotated_frame, (640, 360))

            # Convert the resized frame to a format suitable for Tkinter
            rgb_frame = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk


    def toggle_inference(self):
        # Toggle the inference state
        self.inferencing = not self.inferencing
        self.control_button.config(text="Stop Detection" if self.inferencing else "Start Detection")
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
        Save YOLO inference results to a JSON file in xywhr format, only if OBB values are found.
        Also display the JSON output in the preview box.
        """
        inference_data = []
        # inference_data.append({"image_path": self.image_path})

        for result in results:
            if result.obb is not None and len(result.obb.conf) > 0:
                for i in range(len(result.obb.conf)):
                    class_name = self.model.names[int(result.obb.cls[i])]
                    inference_data.append({
                        "x": float(result.obb.xywhr[i, 0]),
                        "y": float(result.obb.xywhr[i, 1]),
                        "w": float(result.obb.xywhr[i, 2]),
                        "h": float(result.obb.xywhr[i, 3]),
                        "r": float(result.obb.xywhr[i, 4]),
                        "confidence": float(result.obb.conf[i]),
                        "class": class_name
                    })

        # Display the JSON output in the preview box
        self.json_preview.delete(1.0, tk.END)  # Clear the preview box
        if result.obb is not None and len(result.obb.conf) > 0:
            json_output = json.dumps(inference_data, indent=4)
            self.json_preview.insert(tk.END, json_output)  # Insert new JSON data into the box

            # Append to the file if it exists
            if os.path.exists(output_file):
                with open(output_file, "r") as json_file:
                    existing_data = json.load(json_file)
                existing_data.extend(inference_data[1:])  # Skip the first element (image path)
                with open(output_file, "w") as json_file:
                    json.dump(existing_data, json_file, indent=4)
                print(f"Inference results appended to {output_file}")
            else:
                # Create a new file if it doesn't exist
                with open(output_file, "w") as json_file:
                    json.dump(inference_data, json_file, indent=4)
                print(f"Inference results saved to {output_file}")
        else:
            self.json_preview.insert(tk.END, "Nothing Detected or its not in OBB format!.")  # Display a message in the preview box
            print("No OBB data found. Skipping saving to JSON.")

    def release_camera(self):
        """
        Release Camera so that it doesnt get LOCKED forever
        """    
        if self.cap:
            self.cap.release()
            print('camera released due to switching or manual input given!')

    def on_close(self):
        self.release_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    width = 1050
    height = 450
    root.geometry(f"{width}x{height}") 
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
