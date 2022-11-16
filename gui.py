# mode.py must be present in the same folder

import os.path
from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from diffusers import OnnxStableDiffusionPipeline
from modes import render, explore, merge_variations
import threading
import shutil
import os


def start_thread(mode):
    # Disable generate buttons
    gen_button_r["state"] = "disabled"
    gen_button_e["state"] = "disabled"
    gen_button_v["state"] = "disabled"
    t = threading.Thread(target=generate, args=[mode])
    t.start()


def generate(mode):
    if mode == "render":
        # Get values
        values = {"Prompt": prompt_text_r.get(1.0, "end-1c"),
                  "Neg prompt": neg_text_r.get(1.0, "end-1c"),
                  "Inference steps": inf_var_r.get(),
                  "Guidance scale": guidance_var_r.get(),
                  "ETA": eta_var_r.get(),
                  "Seed": seed_text_r.get(),
                  "Output": f"images/render/{output_text_e.get().replace(' ', '_')}",
                  "temp_name": ""}

        # Start generation script
        generated_seed = render(values=values, pipe=pipe, width=512, height=512)

        # Display generated image
        new_image = PhotoImage(file=f"{values['Output']}_{generated_seed}.png")
        canvas_render.create_image(0, 0, anchor=NW, image=new_image)

    elif mode == "explore":
        # Get values
        values = {"Prompt": prompt_text_e.get(1.0, "end-1c"),
                  "Neg prompt": neg_text_e.get(1.0, "end-1c"),
                  "Inference steps": inf_var_e.get(),
                  "Guidance scale": guidance_var_e.get(),
                  "ETA": eta_var_e.get(),
                  "Output": output_text_e.get().replace(" ", "_")}

        # Start generation script
        seeds = explore(values=values, pipe=pipe, width=512, height=512)

        # Display generated image and corresponding seeds
        new_image = PhotoImage(file=f"images/explore/{values['Output']}_{seeds[0]}.png")
        smaller_image = new_image.subsample(2, 2)
        canvas_explore.create_image(0, 0, anchor=NW, image=smaller_image)
	# delete previous seed if any
	text_seed1.delete(0, "end")
        text_seed2.delete(0, "end")
        text_seed3.delete(0, "end")
        text_seed4.delete(0, "end")
	# add new seeds
        text_seed1.insert(0, seeds[0])
        text_seed2.insert(0, seeds[1])
        text_seed3.insert(0, seeds[2])
        text_seed4.insert(0, seeds[3])

    elif mode == "variations":
        # Get values
        values = {"Prompt": prompt_text_v.get(1.0, "end-1c"),
                  "Neg prompt": neg_text_v.get(1.0, "end-1c"),
                  "Inference steps": inf_var_v.get(),
                  "Guidance scale": guidance_var_v.get(),
                  "ETA": eta_var_v.get(),
                  "Seed": seed_text_v.get(),
                  "Output": f"images/variations/{output_text_v.get().replace(' ', '_')}",
                  "temp_name": ""}

        values_varia1 = varia1_text.get().split(",")
        param_varia1 = varia1_box.get()
        values_varia2 = varia2_text.get().split(",")
        param_varia2 = varia2_box.get()

        if not os.path.exists("images/temp"):
            os.mkdir("images/temp")
        else:
            shutil.rmtree("images/temp")
            os.mkdir("images/temp")

        images_list = []  # All images names, in order of generation
        generated_seed = ""
        render_values = values.copy()  # Dict use to run each render with modified info

        # Double loop over the 2 variation parameters to generate all possible images
        for x in values_varia1:
            if param_varia1 in ["Prompt", "Neg prompt"]:
                render_values[param_varia1] = values[param_varia1].replace("REPLACE", x)
            else:
                render_values[param_varia1] = x
            for y in values_varia2:
                if param_varia2 in ["Prompt", "Neg prompt"]:
                    render_values[param_varia2] = values[param_varia2].replace("REPLACE", x)
                else:
                    render_values[param_varia2] = y

                # Set temporary name and save it
                render_values["temp_name"] = f"images/temp/{x}_{y}.png"
                images_list.append(f"images/temp/{x}_{y}.png")

                # Start generation script
                generated_seed = render(values=render_values, pipe=pipe, width=512, height=512)

        # Merge all generated images into one composite image
        merge_variations(images=images_list, varia1=values_varia1, varia2=values_varia2,
                         name=f"{values['Output']}_{generated_seed}.png")

        # Display generated image
        new_image = PhotoImage(file=f"{values['Output']}_{generated_seed}.png")
        dezoom = len(values_varia1) if len(values_varia1) > len(values_varia2) else len(values_varia2)
        smaller_image = new_image.subsample(len(values_varia1)+1, len(values_varia2)+1)
        canvas_variations.create_image(0, 0, anchor=NW, image=smaller_image)

        shutil.rmtree("images/temp")

    # Reactivate generate buttons
    gen_button_r["state"] = "normal"
    gen_button_e["state"] = "normal"
    gen_button_v["state"] = "normal"


if __name__ == "__main__":
    # Check if output folder exists, if not create it
    if not os.path.exists("images"):
        os.mkdir("images")
        os.mkdir("images/render")
        os.mkdir("images/explore")
        os.mkdir("images/variations")

    # Load model
    pipe = OnnxStableDiffusionPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider")

    # Main window
    root = Tk()
    root.title("Stable Chatoune")
    root.geometry('1200x600')
    root.resizable(False, False)

    # Tabs creation
    tabControl = Notebook(root)
    tab_render = Frame(tabControl)
    tab_explore = Frame(tabControl)
    tab_variations = Frame(tabControl)
    tabControl.add(tab_render, text='Render')
    tabControl.add(tab_explore, text='Explore')
    tabControl.add(tab_variations, text='Variations')
    tabControl.pack(expand=1, fill="both")

    ## Render tab widgets ##
    # Create frames
    render_param_frame = Frame(tab_render)
    render_image_frame = Frame(tab_render, padding=(30, 0))
    render_param_frame.pack(side=LEFT)
    render_image_frame.pack(side=RIGHT)

    # Prompt
    prompt_label_r = Label(render_param_frame, text="Prompt", font=("Comic Sans MS", 15))
    prompt_text_r = Text(render_param_frame, width=60, height=5)
    prompt_label_r.grid(column=0, row=1, sticky="W")
    prompt_text_r.grid(column=0, row=2, columnspan=3)

    # Negative prompt
    neg_label_r = Label(render_param_frame, text="Negative Prompt", font=("Comic Sans MS", 15))
    neg_text_r = Text(render_param_frame, width=60, height=5)
    neg_label_r.grid(column=0, row=3, sticky="W")
    neg_text_r.grid(column=0, row=4, columnspan=3)

    # Inference steps
    inf_var_r = IntVar()
    inf_var_r.set(50)
    inf_label_r = Label(render_param_frame, text="Inference steps", font=("Comic Sans MS", 15))
    inf_scale_r = tk.Scale(render_param_frame, variable=inf_var_r, from_=1, to=100, orient=HORIZONTAL, resolution=1,
                           length=300)
    inf_label_r.grid(column=0, row=5, sticky="W")
    inf_scale_r.grid(column=1, row=5, columnspan=2, sticky="W")

    # Guidance scale
    guidance_var_r = DoubleVar()
    guidance_var_r.set(7.5)
    guidance_label_r = Label(render_param_frame, text="Guidance scale", font=("Comic Sans MS", 15))
    guidance_scale_r = tk.Scale(render_param_frame, variable=guidance_var_r, from_=1, to=20, orient=HORIZONTAL,
                                resolution=0.5,
                                length=300)
    guidance_label_r.grid(column=0, row=6, sticky="W")
    guidance_scale_r.grid(column=1, row=6, columnspan=2, sticky="W")

    # ETA
    eta_var_r = DoubleVar()
    eta_label_r = Label(render_param_frame, text="ETA (noise)", font=("Comic Sans MS", 15))
    eta_scale_r = tk.Scale(render_param_frame, variable=eta_var_r, from_=0, to=1, orient=HORIZONTAL, resolution=0.05,
                           length=300)
    eta_label_r.grid(column=0, row=7, sticky="W")
    eta_scale_r.grid(column=1, row=7, columnspan=2, sticky="W")

    # Seed
    seed_bg_r = StringVar()
    seed_label_r = Label(render_param_frame, text="Seed", font=("Comic Sans MS", 15))
    seed_text_r = Entry(render_param_frame, width=30)
    seed_text_r.insert(0, "-1")
    seed_label_r.grid(column=0, row=8, sticky="W")
    seed_text_r.grid(column=1, row=8, sticky="W")

    # Output file
    output_label_r = Label(render_param_frame, text="Image name", font=("Comic Sans MS", 15))
    output_text_r = Entry(render_param_frame, width=30)
    output_text_r.insert(0, "example")
    output_label_r.grid(column=0, row=9, sticky="W")
    output_text_r.grid(column=1, row=9, sticky="W")

    # Generate
    gen_button_r = tk.Button(render_param_frame, text='Generate', command=lambda: start_thread("render"),
                             font=("Comic Sans MS", 30, "bold"), width=15)
    gen_button_r.grid(column=0, row=10, columnspan=3, pady=25)

    # Canvas for image
    canvas_render = Canvas(render_image_frame, width=512, height=512, bg='white')
    canvas_render.pack()

    #############################################################

    ## Explore tab widgets ##
    # Create frames
    explore_param_frame = Frame(tab_explore)
    explore_image_frame = Frame(tab_explore, padding=(30, 0))
    explore_param_frame.pack(side=LEFT)
    explore_image_frame.pack(side=RIGHT)

    # Prompt
    prompt_label_e = Label(explore_param_frame, text="Prompt", font=("Comic Sans MS", 15))
    prompt_text_e = Text(explore_param_frame, width=60, height=5)
    prompt_label_e.grid(column=0, row=1, sticky="W")
    prompt_text_e.grid(column=0, row=2, columnspan=3)

    # Negative prompt
    neg_label_e = Label(explore_param_frame, text="Negative Prompt", font=("Comic Sans MS", 15))
    neg_text_e = Text(explore_param_frame, width=60, height=5)
    neg_label_e.grid(column=0, row=3, sticky="W")
    neg_text_e.grid(column=0, row=4, columnspan=3)

    # Inference steps
    inf_var_e = IntVar()
    inf_var_e.set(20)
    inf_label_e = Label(explore_param_frame, text="Inference steps", font=("Comic Sans MS", 15))
    inf_scale_e = tk.Scale(explore_param_frame, variable=inf_var_e, from_=1, to=100, orient=HORIZONTAL, resolution=1,
                           length=300)
    inf_label_e.grid(column=0, row=5, sticky="W")
    inf_scale_e.grid(column=1, row=5, columnspan=2, sticky="W")

    # Guidance scale
    guidance_var_e = DoubleVar()
    guidance_var_e.set(7.5)
    guidance_label_e = Label(explore_param_frame, text="Guidance scale", font=("Comic Sans MS", 15))
    guidance_scale_e = tk.Scale(explore_param_frame, variable=guidance_var_e, from_=1, to=20, orient=HORIZONTAL,
                                resolution=0.5,
                                length=300)
    guidance_label_e.grid(column=0, row=6, sticky="W")
    guidance_scale_e.grid(column=1, row=6, columnspan=2, sticky="W")

    # ETA
    eta_var_e = DoubleVar()
    eta_label_e = Label(explore_param_frame, text="ETA (noise)", font=("Comic Sans MS", 15))
    eta_scale_e = tk.Scale(explore_param_frame, variable=eta_var_e, from_=0, to=1, orient=HORIZONTAL, resolution=0.05,
                           length=300)
    eta_label_e.grid(column=0, row=7, sticky="W")
    eta_scale_e.grid(column=1, row=7, columnspan=2, sticky="W")

    # Seed
    seed_bg_e = StringVar()
    seed_label_e = Label(explore_param_frame, text="Seed", font=("Comic Sans MS", 15))
    seed_text_e = Entry(explore_param_frame, width=30, background="grey", state="disabled")
    seed_text_e.insert(0, "Four random seeds")
    seed_label_e.grid(column=0, row=8, sticky="W")
    seed_text_e.grid(column=1, row=8, sticky="W")

    # Output file
    output_label_e = Label(explore_param_frame, text="Image name", font=("Comic Sans MS", 15))
    output_text_e = Entry(explore_param_frame, width=30)
    output_text_e.insert(0, "example")
    output_label_e.grid(column=0, row=9, sticky="W")
    output_text_e.grid(column=1, row=9, sticky="W")

    # Generate
    gen_button_e = tk.Button(explore_param_frame, text='Generate', command=lambda: start_thread("explore"),
                             font=("Comic Sans MS", 30, "bold"), width=15)
    gen_button_e.grid(column=0, row=10, columnspan=3, pady=25)

    # Canvas for image + text zones for seeds
    text_seed1 = Entry(explore_image_frame)
    text_seed2 = Entry(explore_image_frame)
    text_seed3 = Entry(explore_image_frame)
    text_seed4 = Entry(explore_image_frame)
    canvas_explore = Canvas(explore_image_frame, width=512, height=512, bg='white')
    text_seed1.grid(column=0, row=0)
    text_seed2.grid(column=1, row=0)
    text_seed3.grid(column=0, row=2)
    text_seed4.grid(column=1, row=2)
    canvas_explore.grid(column=0, row=1, columnspan=2)

    ## Variations tab widgets ##
    # Create Frames
    variations_param_frame = Frame(tab_variations)
    variations_image_frame = Frame(tab_variations, padding=(30, 0))
    variations_param_frame.pack(side=LEFT)
    variations_image_frame.pack(side=RIGHT)

    # Prompt
    prompt_label_v = Label(variations_param_frame, text="Prompt", font=("Comic Sans MS", 15))
    prompt_text_v = Text(variations_param_frame, width=70, height=5)
    prompt_label_v.grid(column=0, row=1, sticky="W")
    prompt_text_v.grid(column=0, row=2, columnspan=3)

    # Negative prompt
    neg_label_v = Label(variations_param_frame, text="Negative Prompt", font=("Comic Sans MS", 15))
    neg_text_v = Text(variations_param_frame, width=70, height=5)
    neg_label_v.grid(column=0, row=3, sticky="W")
    neg_text_v.grid(column=0, row=4, columnspan=3)

    # Inference steps
    inf_var_v = IntVar()
    inf_var_v.set(20)
    inf_label_v = Label(variations_param_frame, text="Inference steps", font=("Comic Sans MS", 15))
    inf_scale_v = tk.Scale(variations_param_frame, variable=inf_var_v, from_=1, to=100, orient=HORIZONTAL, resolution=1,
                           length=300)
    inf_label_v.grid(column=0, row=5, sticky="W")
    inf_scale_v.grid(column=1, row=5, columnspan=2, sticky="W")

    # Guidance scale
    guidance_var_v = DoubleVar()
    guidance_var_v.set(7.5)
    guidance_label_v = Label(variations_param_frame, text="Guidance scale", font=("Comic Sans MS", 15))
    guidance_scale_v = tk.Scale(variations_param_frame, variable=guidance_var_v, from_=1, to=20, orient=HORIZONTAL,
                                resolution=0.5, length=300)
    guidance_label_v.grid(column=0, row=6, sticky="W")
    guidance_scale_v.grid(column=1, row=6, columnspan=2, sticky="W")

    # ETA
    eta_var_v = DoubleVar()
    eta_label_v = Label(variations_param_frame, text="ETA (noise)", font=("Comic Sans MS", 15))
    eta_scale_v = tk.Scale(variations_param_frame, variable=eta_var_v, from_=0, to=1, orient=HORIZONTAL,
                           resolution=0.05,
                           length=300)
    eta_label_v.grid(column=0, row=7, sticky="W")
    eta_scale_v.grid(column=1, row=7, columnspan=2, sticky="W")

    # Seed
    seed_bg_v = StringVar()
    seed_label_v = Label(variations_param_frame, text="Seed", font=("Comic Sans MS", 15))
    seed_text_v = Entry(variations_param_frame, width=40, background="grey")
    seed_text_v.insert(0, "-1")
    seed_label_v.grid(column=0, row=8, sticky="W")
    seed_text_v.grid(column=1, row=8, sticky="W")

    # Output file
    output_label_v = Label(variations_param_frame, text="Image name", font=("Comic Sans MS", 15))
    output_text_v = Entry(variations_param_frame, width=40)
    output_text_v.insert(0, "example")
    output_label_v.grid(column=0, row=9, sticky="W")
    output_text_v.grid(column=1, row=9, sticky="W")

    # Variation 1
    varia1_box = Combobox(variations_param_frame, state="readonly", values=["Prompt", "Neg prompt", "Inference steps",
                                                                            "Guidance scale", "ETA"])
    varia1_box.current(0)
    varia1_text = Entry(variations_param_frame, width=40)
    varia1_box.grid(column=0, row=10, sticky="SW", pady=5)
    varia1_text.grid(column=1, row=10, columnspan=3, sticky="W")

    # Variation 2
    varia2_box = Combobox(variations_param_frame, state="readonly", values=["Prompt", "Neg prompt", "Inference steps",
                                                                            "Guidance scale", "ETA"])
    varia2_box.current(2)
    varia2_text = Entry(variations_param_frame, width=40)
    varia2_box.grid(column=0, row=11, sticky="W")
    varia2_text.grid(column=1, row=11, columnspan=3, sticky="W")

    # Generate
    gen_button_v = tk.Button(variations_param_frame, text='Generate', command=lambda: start_thread("variations"),
                             font=("Comic Sans MS", 30, "bold"), width=15)
    gen_button_v.grid(column=0, row=12, columnspan=3, pady=25)

    # Canvas for image
    canvas_variations = Canvas(variations_image_frame, width=512, height=512, bg='white')
    canvas_variations.pack()

    root.mainloop()
