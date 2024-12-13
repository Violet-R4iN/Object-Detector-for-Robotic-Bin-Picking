import tkinter as tk
from tkinter import ttk
import cv2
from ultralytics import YOLO
from threading import Thread
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yolov11-GUI")

        # Initialize variables
        self.cameras = self.get_available_cameras()
        self.current_camera_index = 1  # Default to the second camera
        self.running = False
        self.inferencing = False
        self.cap = None
        self.weight_path = "bestV3-OBB.pt"
        self.model = YOLO(self.weight_path)

        # Create UI
        self.create_ui()
        self.select_camera(self.current_camera_index)

    def create_ui(self):
        # Frame for top controls
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10, side=tk.TOP)

        # Combobox to select the camera
        self.camera_selector = ttk.Combobox(top_frame, values=self.cameras, state="readonly")
        self.camera_selector.pack(side=tk.LEFT, padx=5)
        self.camera_selector.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # Set default selection
        if self.cameras:
            self.camera_selector.current(self.current_camera_index)
        else:
            self.camera_selector.set("No Cameras Found")

        self.camera_selector.bind("<<ComboboxSelected>>", self.on_camera_selected)


        # Button to start/stop inference
        self.control_button = tk.Button(top_frame, text="Start Inference", command=self.toggle_inference)
        self.control_button.pack(side=tk.LEFT, padx=5)

        # Weight selector
        self.weight_label = tk.Label(top_frame, text=f"Weight: {self.weight_path}")
        self.weight_label.pack(side=tk.LEFT, padx=5)
        self.weight_button = tk.Button(top_frame, text="Select Weight", command=self.select_weight)
        self.weight_button.pack(side=tk.LEFT, padx=5)

        # Canvas to show the camera feed
        self.canvas = tk.Canvas(self.root, width=640, height=360)  # Adjusted to 16:9 aspect ratio
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

    def get_available_cameras(self):
        """
        Check for available cameras using port testing.
        """
        available_ports, working_ports, non_working_ports = self.list_ports()
        
        # Combine working and available ports
        camera_ports = working_ports
        
        # Convert port numbers to camera names
        cameras = [f"Camera {port}" for port in camera_ports]
        return cameras
    

    def list_ports(self):
        """
        Test the ports and returns a tuple with the available ports and the ones that are working.
        """
        non_working_ports = []
        dev_port = 0
        working_ports = []
        available_ports = []
        while len(non_working_ports) < 6:  # if there are more than 5 non working ports stop the testing. 
            camera = cv2.VideoCapture(dev_port, cv2.CAP_DSHOW)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    print(f"Camera {dev_port} is working and reads images ({h} x {w})")
                    working_ports.append(dev_port)
                else:
                    print(f"Camera {dev_port} for camera ({h} x {w}) is present but does not read.")
                    available_ports.append(dev_port)
                camera.release()
            dev_port += 1
        return available_ports, working_ports, non_working_ports
    
    
    def on_camera_selected(self, event):
        # Handle camera selection change
        selected_index = self.camera_selector.current()
        if selected_index != self.current_camera_index:
            self.select_camera(selected_index)

    def select_camera(self, index):
        # Stop current camera if running
        self.running = False
        if self.cap:
            self.cap.release()

        self.current_camera_index = index
        self.cap = cv2.VideoCapture(self.current_camera_index)  
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.current_camera_index}")
            return

        self.running = True
        Thread(target=self.update_frame, daemon=True).start()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            if self.inferencing:
                # Run YOLO inference and annotate the frame
                results = self.model(frame, conf=0.8)
                annotated_frame = results[0].plot(font_size=0.5)
            else:
                annotated_frame = frame

            # Convert the frame to a format suitable for Tkinter
            frame_resized = cv2.resize(annotated_frame, (640, 360))  # Resize to 16:9 aspect ratio
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the frame on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

    def toggle_inference(self):
        # Toggle the inference state
        self.inferencing = not self.inferencing
        self.control_button.config(text="Stop Inference" if self.inferencing else "Start Inference")

    def select_weight(self):
        # Select a new weight file
        from tkinter.filedialog import askopenfilename
        weight_file = askopenfilename(filetypes=[("YOLO Weight Files", "*.pt")])
        if weight_file:
            self.weight_path = weight_file
            self.weight_label.config(text=f"Weight: {self.weight_path}")
            self.model = YOLO(self.weight_path)

    def on_close(self):
        # Stop the application
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
