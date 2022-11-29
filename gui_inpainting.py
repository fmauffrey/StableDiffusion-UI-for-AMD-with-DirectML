import os.path
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import tkinter as tk
from diffusers import OnnxStableDiffusionInpaintPipeline
import threading
from modes import inpainting
from PIL import ImageGrab, Image


# Function for opening the file explorer window
def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                          filetypes=(("Image png", "*.png*"), ("all files", "*.*")))

    # Update text input file
    input_im_text_r.config(state="normal")
    input_im_text_r.delete(0, "end")
    input_im_text_r.insert(0, filename)
    input_im_text_r.config(state="disabled")

    # Resize image
    img = Image.open(filename)
    img_resized = img.resize((512, 512))
    img_resized.save("images/inpaint_input.png")

    # Load in Canvas
    image = tk.PhotoImage(file="images/inpaint_input.png")
    canvas_background.create_image(0, 0, anchor=NW, image=image)
    image_input.append(image)


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas_background.create_oval(x1, y1, x2, y2, fill=color.get(), outline=color.get(), width=size_value.get())


def clear():
    # ReLoad in Canvas
    canvas_background.create_image(0, 0, anchor=NW, image=image_input[-1])


def start_thread(mode):
    # Disable generate buttons
    gen_button_r["state"] = "disabled"
    t = threading.Thread(target=generate)
    t.start()


def generate():
    # Save mask image
    x = Canvas.winfo_rootx(canvas_background)
    y = Canvas.winfo_rooty(canvas_background)
    w = Canvas.winfo_width(canvas_background)
    h = Canvas.winfo_height(canvas_background)

    img = ImageGrab.grab((x, y, x + w, y + h))
    pixels = img.load()  # create the pixel map

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixels[i, j] != (255, 255, 255):
                pixels[i, j] = (0, 0, 0)

    img.save("images/mask.png")

    # Get values
    values = {"Prompt": prompt_text_r.get(1.0, "end-1c"),
              "Guidance scale": guidance_var_r.get(),
              "Strength": strength_scale_r.get(),
              "Image": input_im_text_r.get(),
              "Output": f"images/inpainting/{output_text_r.get().replace(' ', '_')}"}

    # Start generation script
    inpainting(values=values, pipe=pipe)

    # Display generated image
    new_image = PhotoImage(file=f"{values['Output']}.png")
    canvas_render.create_image(0, 0, anchor=NW, image=new_image)

    # Reactivate generate buttons
    gen_button_r["state"] = "normal"


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

    # Main window
    root = Tk()
    root.title("Stable Chatoune")
    root.geometry('1600x600')
    root.resizable(False, False)

    ## Render tab widgets ##
    # Create frames
    top_frame = Frame(root)
    bot_frame = Frame(root)
    inpaint_param_frame = Frame(top_frame)
    inpaint_image_frame_left = Frame(top_frame, padding=(30, 0))
    inpaint_image_frame_right = Frame(top_frame, padding=(30, 0))
    button_frame = Frame(bot_frame)
    top_frame.pack(side=TOP)
    bot_frame.pack(side=BOTTOM)
    inpaint_param_frame.pack(side=LEFT)
    inpaint_image_frame_left.pack(side=LEFT)
    inpaint_image_frame_right.pack(side=LEFT)
    button_frame.pack()

    # Prompt
    prompt_label_r = Label(inpaint_param_frame, text="Prompt", font=("Comic Sans MS", 15))
    prompt_text_r = Text(inpaint_param_frame, width=60, height=5)
    prompt_label_r.grid(column=0, row=1, sticky="W")
    prompt_text_r.grid(column=0, row=2, columnspan=3)

    # Guidance scale
    guidance_var_r = DoubleVar()
    guidance_var_r.set(7.5)
    guidance_label_r = Label(inpaint_param_frame, text="Guidance scale", font=("Comic Sans MS", 15))
    guidance_scale_r = tk.Scale(inpaint_param_frame, variable=guidance_var_r, from_=1, to=20, orient=HORIZONTAL,
                                resolution=0.5, length=300)
    guidance_label_r.grid(column=0, row=6, sticky="W")
    guidance_scale_r.grid(column=1, row=6, columnspan=2, sticky="W")

    # Strength
    strength_value_r = DoubleVar()
    strength_value_r.set(0.75)
    strength_label_r = Label(inpaint_param_frame, text="Strength", font=("Comic Sans MS", 15))
    strength_scale_r = tk.Scale(inpaint_param_frame, variable=strength_value_r, from_=0, to=1, orient=HORIZONTAL,
                                resolution=0.05, length=300)
    strength_label_r.grid(column=0, row=7, sticky="W")
    strength_scale_r.grid(column=1, row=7, columnspan=2, sticky="W")

    # Output file
    output_label_r = Label(inpaint_param_frame, text="Image name", font=("Comic Sans MS", 15))
    output_text_r = Entry(inpaint_param_frame, width=30)
    output_text_r.insert(0, "example")
    output_label_r.grid(column=0, row=8, sticky="W")
    output_text_r.grid(column=1, row=8, sticky="W")

    # Input file
    input_im_label_r = Label(inpaint_param_frame, text="Input image", font=("Comic Sans MS", 15))
    input_im_text_r = Entry(inpaint_param_frame, width=50, state="disabled")
    input_im_text_r.insert(0, "")
    input_im_label_r.grid(column=0, row=9, sticky="W")
    input_im_text_r.grid(column=1, row=9, sticky="W")

    # Generate
    gen_button_r = tk.Button(inpaint_param_frame, text='Generate', command=lambda: start_thread("inpainting"),
                             font=("Comic Sans MS", 30, "bold"), width=15)
    gen_button_r.grid(column=0, row=10, columnspan=3, pady=25)

    # Canvas for image input
    canvas_background = Canvas(inpaint_image_frame_left, width=512, height=512, bg='white')
    canvas_background.grid(column=0, row=0)

    # Canvas buttons
    load_btn = tk.Button(button_frame, text="Load image", command=browseFiles)
    clear_btn = tk.Button(button_frame, text="Clear mask", command=clear)
    size_value = tk.IntVar(value=25)
    paint_size = tk.Spinbox(button_frame, from_=2, to=50, textvariable=size_value)
    canvas_background.bind("<B1-Motion>", paint)
    color = StringVar()
    color.set("white")
    color_btn1 = Radiobutton(button_frame, text="white", variable=color, value="white", state="normal")
    color_btn1.pack(side=LEFT)
    color_btn2 = Radiobutton(button_frame, text="black", variable=color, value="black")
    color_btn2.pack(side=LEFT)
    load_btn.pack(side=LEFT)
    clear_btn.pack(side=LEFT)
    paint_size.pack(side=LEFT, padx=10)

    # Canvas for image output
    canvas_render = Canvas(inpaint_image_frame_right, width=512, height=512, bg='white')
    canvas_render.pack()

    root.mainloop()
