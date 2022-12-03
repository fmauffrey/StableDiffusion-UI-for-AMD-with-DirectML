import os.path
import tkinter as tk
from tkinter import filedialog
from diffusers import OnnxStableDiffusionInpaintPipeline
import threading
from PIL import ImageGrab, Image
import customtkinter
import sys
import io
import time
import re


class App(customtkinter.CTkFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_input = []

        # Redirecting output to variable
        self.temp_out = io.StringIO()
        sys.stderr = self.temp_out

        # Check if output folder exists, if not create it
        if not os.path.exists("images"):
            os.mkdir("images")
        if not os.path.exists("images/inpainting"):
            os.mkdir("images/inpainting")

        # Graphical parameters
        customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

        # Load model
        self.pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained("./stable_diffusion_onnx_inpainting",
                                                                       provider="DmlExecutionProvider", revision="onnx",
                                                                       safety_checker=None)

        # Main window
        parent.title("Stable UI Inpainting")
        parent.geometry('1600x800')
        parent.resizable(False, False)

        # Prompt
        self.prompt_label = customtkinter.CTkLabel(self, text="Prompt", text_font=("Arial", 15))
        self.prompt_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.prompt_text = customtkinter.CTkTextbox(self, width=500, height=100)
        self.prompt_label.place(x=50, y=20)
        self.prompt_text.place(x=50, y=50)

        # Guidance scale
        self.guidance_var = tk.DoubleVar()
        self.guidance_var.set(7.5)
        self.guidance_label = customtkinter.CTkLabel(self, text="Guidance scale", text_font=("Arial", 15))
        self.guidance_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.guidance_scale = customtkinter.CTkSlider(self, variable=self.guidance_var, from_=1, to=20,
                                                      number_of_steps=38, width=200, command=self.slider_event_guidance)
        self.guidance_scale.set(12)
        self.guidance_counter_var = tk.StringVar(value="12.0")
        self.guidance_counter = customtkinter.CTkLabel(self, textvariable=self.guidance_counter_var,
                                                       text_font=("Arial", 15))
        self.guidance_counter.text_label.place(relx=0, rely=0.5, anchor="w")
        self.guidance_label.place(x=600, y=60)
        self.guidance_scale.place(x=750, y=65)
        self.guidance_counter.place(x=960, y=60)

        # Strength
        self.strength_value = tk.DoubleVar()
        self.strength_value.set(0.75)
        self.strength_label = customtkinter.CTkLabel(self, text="Strength", text_font=("Arial", 15))
        self.strength_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.strength_scale = customtkinter.CTkSlider(self, variable=self.strength_value, from_=0, to=1,
                                                      number_of_steps=20, width=200, command=self.slider_event_strength)
        self.strength_scale.set(0.75)
        self.strength_counter_var = tk.StringVar(value="0.75")
        self.strength_counter = customtkinter.CTkLabel(self, textvariable=self.strength_counter_var,
                                                       text_font=("Arial", 15))
        self.strength_counter.text_label.place(relx=0, rely=0.5, anchor="w")
        self.strength_label.place(x=600, y=100)
        self.strength_scale.place(x=750, y=105)
        self.strength_counter.place(x=960, y=100)

        # Output file
        self.output_label = customtkinter.CTkLabel(self, text="Image name", text_font=("Arial", 15))
        self.output_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.output_text = customtkinter.CTkEntry(self, width=280)
        self.output_text.insert(0, "example")
        self.output_label.place(x=1050, y=100)
        self.output_text.place(x=1200, y=100)

        # Input file
        self.input_im_label = customtkinter.CTkLabel(self, text="Input image", text_font=("Arial", 15))
        self.input_im_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.input_im_text = customtkinter.CTkEntry(self, width=280, state="disabled")
        self.input_im_text.insert(0, "")
        self.load_btn = customtkinter.CTkButton(self, text="Import", command=self.browse_files, text_font=("Arial", 10),
                                                width=50)
        self.input_im_label.place(x=1050, y=60)
        self.input_im_text.place(x=1200, y=60)
        self.load_btn.place(x=1500, y=60)

        # Generate
        self.gen_button = customtkinter.CTkButton(self, text='Generate',
                                                  command=lambda: self.start_thread(),
                                                  text_font=("Arial", 30, "bold"), width=300, height=80)
        self.gen_button.place(x=650, y=450)

        # Generation speed
        self.speed_var = tk.StringVar()
        self.speed_var.set("")
        self.speed_label = customtkinter.CTkLabel(self, textvariable=self.speed_var, width=100, text_font=("Arial", 10))
        self.speed_label.place(x=750, y=540)

        # Progress bar
        self.prog_bar = customtkinter.CTkProgressBar(master=self, width=300)
        self.prog_bar.set(0)
        self.prog_bar.place(x=650, y=535)

        # Canvas for image input
        self.canvas_background = tk.Canvas(self, width=512, height=512, bg='white')
        self.canvas_background.place(x=100, y=250)

        # Canvas buttons
        size_value = tk.IntVar(value=25)
        self.canvas_background.bind("<B1-Motion>", self.paint)
        self.color = tk.StringVar()
        self.color.set("white")
        self.color_btn1 = customtkinter.CTkRadioButton(self, text="white", variable=self.color, value="white",
                                                       state="normal",
                                                       text_font=("Arial", 10))
        self.color_btn2 = customtkinter.CTkRadioButton(self, text="black", variable=self.color, value="black",
                                                       text_font=("Arial", 10))
        self.color_btn1.place(x=100, y=210)
        self.color_btn2.place(x=170, y=210)

        # Paint brush size
        self.brush_value = tk.DoubleVar()
        self.brush_value.set(30)
        self.brush_label = customtkinter.CTkLabel(self, text="Brush size", text_font=("Arial", 10))
        self.brush_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.brush_scale = customtkinter.CTkSlider(self, variable=self.brush_value, from_=1, to=50,
                                                   number_of_steps=49, width=150, command=self.slider_event_brush)
        self.brush_scale.set(30)
        self.brush_counter_var = tk.StringVar(value="30.0")
        self.brush_counter = customtkinter.CTkLabel(self, textvariable=self.brush_counter_var, text_font=("Arial", 10))
        self.brush_counter.text_label.place(relx=0, rely=0.5, anchor="w")
        self.brush_label.place(x=250, y=207)
        self.brush_scale.place(x=315, y=215)
        self.brush_counter.place(x=465, y=208)

        # Clear button
        self.clear_btn = customtkinter.CTkButton(self, text="Clear mask", command=self.clear, width=50)
        self.clear_btn.place(x=530, y=205)

        # Canvas for image output
        self.canvas_render = tk.Canvas(self, width=512, height=512, bg='white')
        self.canvas_render.place(x=988, y=250)

    # Function for opening the file explorer window
    def browse_files(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                              filetypes=(("Image png", "*.png*"), ("all files", "*.*")))

        # Update text input file
        self.input_im_text.configure(state="normal")
        self.input_im_text.delete(0, "end")
        self.input_im_text.insert(0, filename)
        self.input_im_text.configure(state="disabled")

        # Resize image
        img = Image.open(filename)
        img_resized = img.resize((512, 512))
        img_resized.save("images/inpaint_input.png")

        # Load in Canvas
        image = tk.PhotoImage(file="images/inpaint_input.png")
        self.canvas_background.create_image(0, 0, anchor="nw", image=image)
        self.image_input.append(image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas_background.create_oval(x1, y1, x2, y2, fill=self.color.get(), outline=self.color.get(),
                                           width=self.brush_scale.get())

    def clear(self):
        # ReLoad in Canvas
        self.canvas_background.create_image(0, 0, anchor="nw", image=self.image_input[-1])

    def slider_event_guidance(self, value):
        self.guidance_counter_var.set(str(value))

    def slider_event_strength(self, value):
        rounded_value = round(value, 2)
        self.strength_counter_var.set(str(rounded_value))

    def slider_event_brush(self, value):
        self.brush_counter_var.set(str(value))

    def stdout_listening(self):
        prog = 0
        while prog < 100:
            output = self.temp_out.getvalue()
            try:
                # Get % of progress from output
                prog_text = re.findall(r"\d{1,3}(?=%)", output)
                prog = int(prog_text[-1])
                self.prog_bar.set(prog / 100)
                it_text = re.findall(r"\d\.\d{1,2}s/it|\d\.\d{1,2}its/s", output)
                it = it_text[-1]
                self.speed_var.set(it)
            except:
                pass
            time.sleep(0.5)

        # Recreate temporary sdterr
        self.temp_out = io.StringIO()
        sys.stderr = self.temp_out

        # Reinitialize speed variable
        self.speed_var.set("")

    def start_thread(self):
        # Disable generate buttons
        self.gen_button.configure(state="disabled")
        t = threading.Thread(target=self.generate)
        l = threading.Thread(target=self.stdout_listening)
        t.start()
        l.start()

    def generate(self):
        # Save mask image
        x = tk.Canvas.winfo_rootx(self.canvas_background)
        y = tk.Canvas.winfo_rooty(self.canvas_background)
        w = tk.Canvas.winfo_width(self.canvas_background)
        h = tk.Canvas.winfo_height(self.canvas_background)

        img = ImageGrab.grab((x, y, x + w, y + h))
        pixels = img.load()  # create the pixel map

        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixels[i, j] != (255, 255, 255):
                    pixels[i, j] = (0, 0, 0)

        img.save("images/mask.png")

        # Get values
        values = {"Prompt": self.prompt_text.textbox.get("0.0", "end"),
                  "Guidance scale": self.guidance_var.get(),
                  "Strength": self.strength_scale.get(),
                  "Image": self.input_im_text.get(),
                  "Output": f"images/inpainting/{self.output_text.get().replace(' ', '_')}"}

        # Start generation script
        image = \
            self.pipe(prompt=values["Prompt"], image=Image.open(values["Image"]),
                      mask_image=Image.open("images/mask.png"),
                      guidance_scale=values["Guidance scale"]).images[0]
        image.save(f"{values['Output']}.png")

        # Display generated image
        new_image = tk.PhotoImage(file=f"{values['Output']}.png")
        self.canvas_render.create_image(0, 0, anchor="nw", image=new_image)

        # Reactivate generate buttons
        self.gen_button.configure(state="normal")


if __name__ == "__main__":
    root = customtkinter.CTk()
    App(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
