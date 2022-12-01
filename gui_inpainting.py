import os.path
import tkinter as tk
from tkinter import filedialog
from diffusers import OnnxStableDiffusionInpaintPipeline
import threading
from modes import inpainting
from PIL import ImageGrab, Image
import customtkinter


# Function for opening the file explorer window
def browse_files():
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                          filetypes=(("Image png", "*.png*"), ("all files", "*.*")))

    # Update text input file
    input_im_text.configure(state="normal")
    input_im_text.delete(0, "end")
    input_im_text.insert(0, filename)
    input_im_text.configure(state="disabled")

    # Resize image
    img = Image.open(filename)
    img_resized = img.resize((512, 512))
    img_resized.save("images/inpaint_input.png")

    # Load in Canvas
    image = tk.PhotoImage(file="images/inpaint_input.png")
    canvas_background.create_image(0, 0, anchor="nw", image=image)
    image_input.append(image)


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas_background.create_oval(x1, y1, x2, y2, fill=color.get(), outline=color.get(), width=brush_scale.get())


def clear():
    # ReLoad in Canvas
    canvas_background.create_image(0, 0, anchor="nw", image=image_input[-1])


def slider_event_guidance(value):
    guidance_counter_var.set(str(value))


def slider_event_strength(value):
    rounded_value = round(value, 2)
    strength_counter_var.set(str(rounded_value))


def slider_event_brush(value):
    brush_counter_var.set(str(value))


def start_thread(mode):
    # Disable generate buttons
    gen_button.configure(state="disabled")
    t = threading.Thread(target=generate)
    t.start()


def generate():
    # Save mask image
    x = tk.Canvas.winfo_rootx(canvas_background)
    y = tk.Canvas.winfo_rooty(canvas_background)
    w = tk.Canvas.winfo_width(canvas_background)
    h = tk.Canvas.winfo_height(canvas_background)

    img = ImageGrab.grab((x, y, x + w, y + h))
    pixels = img.load()  # create the pixel map

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixels[i, j] != (255, 255, 255):
                pixels[i, j] = (0, 0, 0)

    img.save("images/mask.png")

    # Get values
    values = {"Prompt": prompt_text.textbox.get("0.0", "end"),
              "Guidance scale": guidance_var.get(),
              "Strength": strength_scale.get(),
              "Image": input_im_text.get(),
              "Output": f"images/inpainting/{output_text.get().replace(' ', '_')}"}

    # Start generation script
    inpainting(values=values, pipe=pipe)

    # Display generated image
    new_image = tk.PhotoImage(file=f"{values['Output']}.png")
    canvas_render.create_image(0, 0, anchor="nw", image=new_image)

    # Reactivate generate buttons
    gen_button.configure(state="normal")


if __name__ == "__main__":
    image_input = []
    # Check if output folder exists, if not create it
    if not os.path.exists("images"):
        os.mkdir("images")
    if not os.path.exists("images/inpainting"):
        os.mkdir("images/inpainting")

    # Load model
    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained("./stable_diffusion_onnx_inpainting",
                                                              provider="DmlExecutionProvider", revision="onnx",
                                                              safety_checker=None)

    # Graphical parameters
    customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
    customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

    # Main window
    root = customtkinter.CTk()
    root.title("Stable UI Inpainting")
    root.geometry('1600x800')
    root.resizable(False, False)

    # Prompt
    prompt_label = customtkinter.CTkLabel(root, text="Prompt", text_font=("Arial", 15))
    prompt_label.text_label.place(relx=0, rely=0.5, anchor="w")
    prompt_text = customtkinter.CTkTextbox(root, width=500, height=100)
    prompt_label.place(x=50, y=20)
    prompt_text.place(x=50, y=50)

    # Guidance scale
    guidance_var = tk.DoubleVar()
    guidance_var.set(7.5)
    guidance_label = customtkinter.CTkLabel(root, text="Guidance scale", text_font=("Arial", 15))
    guidance_label.text_label.place(relx=0, rely=0.5, anchor="w")
    guidance_scale = customtkinter.CTkSlider(root, variable=guidance_var, from_=1, to=20,
                                             number_of_steps=38, width=200, command=slider_event_guidance)
    guidance_scale.set(12)
    guidance_counter_var = tk.StringVar(value="12.0")
    guidance_counter = customtkinter.CTkLabel(root, textvariable=guidance_counter_var, text_font=("Arial", 15))
    guidance_counter.text_label.place(relx=0, rely=0.5, anchor="w")
    guidance_label.place(x=600, y=60)
    guidance_scale.place(x=750, y=65)
    guidance_counter.place(x=960, y=60)

    # Strength
    strength_value = tk.DoubleVar()
    strength_value.set(0.75)
    strength_label = customtkinter.CTkLabel(root, text="Strength", text_font=("Arial", 15))
    strength_label.text_label.place(relx=0, rely=0.5, anchor="w")
    strength_scale = customtkinter.CTkSlider(root, variable=strength_value, from_=0, to=1,
                                             number_of_steps=20, width=200, command=slider_event_strength)
    strength_scale.set(0.75)
    strength_counter_var = tk.StringVar(value="0.75")
    strength_counter = customtkinter.CTkLabel(root, textvariable=strength_counter_var, text_font=("Arial", 15))
    strength_counter.text_label.place(relx=0, rely=0.5, anchor="w")
    strength_label.place(x=600, y=100)
    strength_scale.place(x=750, y=105)
    strength_counter.place(x=960, y=100)

    # Output file
    output_label = customtkinter.CTkLabel(root, text="Image name", text_font=("Arial", 15))
    output_label.text_label.place(relx=0, rely=0.5, anchor="w")
    output_text = customtkinter.CTkEntry(root, width=280)
    output_text.insert(0, "example")
    output_label.place(x=1050, y=100)
    output_text.place(x=1200, y=100)

    # Input file
    input_im_label = customtkinter.CTkLabel(root, text="Input image", text_font=("Arial", 15))
    input_im_label.text_label.place(relx=0, rely=0.5, anchor="w")
    input_im_text = customtkinter.CTkEntry(root, width=280, state="disabled")
    input_im_text.insert(0, "")
    load_btn = customtkinter.CTkButton(root, text="Import", command=browse_files, text_font=("Arial", 10), width=50)
    input_im_label.place(x=1050, y=60)
    input_im_text.place(x=1200, y=60)
    load_btn.place(x=1500, y=60)

    # Generate
    gen_button = customtkinter.CTkButton(root, text='Generate',
                                         command=lambda: start_thread("inpainting"),
                                         text_font=("Arial", 30, "bold"), width=300, height=80)
    gen_button.place(x=650, y=450)

    # Progress bar
    prog_bar = customtkinter.CTkProgressBar(master=root, width=300)
    prog_bar.set(0)
    prog_bar.place(x=650, y=535)

    # Canvas for image input
    canvas_background = tk.Canvas(root, width=512, height=512, bg='white')
    canvas_background.place(x=100, y=250)

    # Canvas buttons
    size_value = tk.IntVar(value=25)
    canvas_background.bind("<B1-Motion>", paint)
    color = tk.StringVar()
    color.set("white")
    color_btn1 = customtkinter.CTkRadioButton(root, text="white", variable=color, value="white", state="normal",
                                              text_font=("Arial", 10))
    color_btn2 = customtkinter.CTkRadioButton(root, text="black", variable=color, value="black",
                                              text_font=("Arial", 10))
    color_btn1.place(x=100, y=210)
    color_btn2.place(x=170, y=210)

    # Paint brush size
    brush_value = tk.DoubleVar()
    brush_value.set(30)
    brush_label = customtkinter.CTkLabel(root, text="Brush size", text_font=("Arial", 10))
    brush_label.text_label.place(relx=0, rely=0.5, anchor="w")
    brush_scale = customtkinter.CTkSlider(root, variable=brush_value, from_=1, to=50,
                                          number_of_steps=49, width=150, command=slider_event_brush)
    brush_scale.set(30)
    brush_counter_var = tk.StringVar(value="30.0")
    brush_counter = customtkinter.CTkLabel(root, textvariable=brush_counter_var, text_font=("Arial", 10))
    brush_counter.text_label.place(relx=0, rely=0.5, anchor="w")
    brush_label.place(x=250, y=207)
    brush_scale.place(x=315, y=215)
    brush_counter.place(x=465, y=208)

    # Clear button
    clear_btn = customtkinter.CTkButton(root, text="Clear mask", command=clear, width=50)
    clear_btn.place(x=530, y=205)

    # Canvas for image output
    canvas_render = tk.Canvas(root, width=512, height=512, bg='white')
    canvas_render.place(x=988, y=250)

    root.mainloop()
