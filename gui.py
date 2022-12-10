import tkinter as tk
from diffusers import OnnxStableDiffusionPipeline
import threading
import shutil
import os
import sys
import io
import re
import time
import customtkinter
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from diffusers import (DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                       EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler)


class App(customtkinter.CTkFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check if output folder exists, if not create it
        if not os.path.exists("images"):
            os.mkdir("images")
            os.mkdir("images/render")
            os.mkdir("images/explore")
            os.mkdir("images/variations")

        # Pipe and scheduler variables
        self.scheduler = None
        self.pipe = None

        # Graphical parameters
        customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green
        title_size = 15
        text_size = 12

        # Variable to store image for canvas
        self.image = None
        self.smaller_image = None

        # Redirecting output to variable
        self.temp_out = io.StringIO()
        sys.stderr = self.temp_out

        # Main window
        parent.title("Stable UI")
        parent.geometry('1600x650')
        parent.resizable(False, False)

        # Frames
        self.frame_scheduler_1 = customtkinter.CTkFrame(parent, width=800, height=50, corner_radius=0)
        self.frame_scheduler_1.pack_propagate(False)
        self.frame_scheduler_2 = customtkinter.CTkFrame(parent, width=800, height=50, corner_radius=0)
        self.frame_scheduler_2.pack_propagate(False)
        self.frame_mode = customtkinter.CTkFrame(parent, width=800, height=50, corner_radius=0)
        self.frame_mode.pack_propagate(False)
        self.frame_params = customtkinter.CTkFrame(parent, width=800, height=300, corner_radius=10)
        self.frame_params.grid_propagate(False)
        self.frame_scheduler_1.place(x=20, y=50)
        self.frame_scheduler_2.place(x=20, y=100)
        self.frame_mode.place(x=20, y=200)
        self.frame_params.place(x=20, y=310)

        # Scheduler options
        self.scheduler_label = customtkinter.CTkLabel(self, text="Scheduler", text_font=("Arial", title_size))
        self.scheduler_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.scheduler_var = tk.StringVar()
        self.scheduler_button_1 = customtkinter.CTkRadioButton(self.frame_scheduler_1, text="DDPMScheduler",
                                                               variable=self.scheduler_var,
                                                               value="DDPMScheduler", command=self.scheduler_choice,
                                                               text_font=("Arial", text_size))
        self.scheduler_button_2 = customtkinter.CTkRadioButton(self.frame_scheduler_1, text="DDIMScheduler",
                                                               variable=self.scheduler_var,
                                                               value="DDIMScheduler", command=self.scheduler_choice,
                                                               text_font=("Arial", text_size))
        self.scheduler_button_3 = customtkinter.CTkRadioButton(self.frame_scheduler_1, text="PNDMScheduler",
                                                               variable=self.scheduler_var,
                                                               value="PNDMScheduler", command=self.scheduler_choice,
                                                               text_font=("Arial", text_size))
        self.scheduler_button_4 = customtkinter.CTkRadioButton(self.frame_scheduler_1, text="LMSDiscreteScheduler",
                                                               variable=self.scheduler_var,
                                                               value="LMSDiscreteScheduler",
                                                               command=self.scheduler_choice,
                                                               text_font=("Arial", text_size))
        self.scheduler_button_5 = customtkinter.CTkRadioButton(self.frame_scheduler_2, text="EulerDiscreteScheduler",
                                                               variable=self.scheduler_var,
                                                               value="EulerDiscreteScheduler",
                                                               command=self.scheduler_choice,
                                                               text_font=("Arial", text_size))
        self.scheduler_button_6 = customtkinter.CTkRadioButton(self.frame_scheduler_2,
                                                               text="EulerAncestralDiscreteScheduler",
                                                               variable=self.scheduler_var,
                                                               value="EulerAncestralDiscreteScheduler",
                                                               command=self.scheduler_choice,
                                                               text_font=("Arial", text_size))
        self.scheduler_button_7 = customtkinter.CTkRadioButton(self.frame_scheduler_2,
                                                               text="DPMSolverMultistepScheduler",
                                                               variable=self.scheduler_var,
                                                               value="DPMSolverMultistepScheduler",
                                                               command=self.scheduler_choice,
                                                               text_font=("Arial", text_size))
        self.scheduler_label.place(x=20, y=20)
        for x in [self.scheduler_button_1, self.scheduler_button_2, self.scheduler_button_3, self.scheduler_button_4,
                  self.scheduler_button_5,
                  self.scheduler_button_6, self.scheduler_button_7]:
            x.pack(expand=True, padx=10, pady=10, side="left", anchor="w")

        # Mode options
        self.mode_label = customtkinter.CTkLabel(self, text="Mode", text_font=("Arial", title_size))
        self.mode_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.mode_var = tk.StringVar()
        self.render_button = customtkinter.CTkRadioButton(self.frame_mode, text="Render", variable=self.mode_var,
                                                          value="Render",
                                                          text_font=("Arial", text_size), command=self.mode_changer)
        self.explore_button = customtkinter.CTkRadioButton(self.frame_mode, text="Explore", variable=self.mode_var,
                                                           value="Explore",
                                                           text_font=("Arial", text_size), command=self.mode_changer)
        self.variations_button = customtkinter.CTkRadioButton(self.frame_mode, text="Variations",
                                                              variable=self.mode_var,
                                                              value="Variations", text_font=("Arial", text_size),
                                                              command=self.mode_changer)
        self.mode_label.place(x=20, y=170)
        for x in [self.render_button, self.explore_button, self.variations_button]:
            x.pack(expand=True, padx=10, pady=10, side="left", anchor="w")

        # Parameters title
        self.param_title = customtkinter.CTkLabel(self, text="Parameters", text_font=("Arial", title_size))
        self.param_title.text_label.place(relx=0, rely=0.5, anchor="w")
        self.param_title.place(x=20, y=280)

        # Prompt
        self.prompt_label = customtkinter.CTkLabel(self.frame_params, text="Prompt", text_font=("Arial", title_size))
        self.prompt_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.prompt_text = customtkinter.CTkTextbox(self.frame_params, width=350, height=100)
        self.prompt_label.place(x=25, y=10)
        self.prompt_text.place(x=25, y=40)

        # Negative prompt
        self.neg_label = customtkinter.CTkLabel(self.frame_params, text="Negative Prompt",
                                                text_font=("Arial", title_size),
                                                width=200)
        self.neg_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.neg_text = customtkinter.CTkTextbox(self.frame_params, width=350, height=100)
        self.neg_label.place(x=425, y=10)
        self.neg_text.place(x=425, y=40)

        # Inference steps
        self.inf_label = customtkinter.CTkLabel(self.frame_params, text="Inference steps",
                                                text_font=("Arial", title_size))
        self.inf_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.inf_var = tk.IntVar()
        self.inf_var.set(50)
        self.inf_scale = customtkinter.CTkSlider(self.frame_params, variable=self.inf_var, from_=1, to=100,
                                                 number_of_steps=99,
                                                 width=150,
                                                 command=self.slider_event_inference)
        self.inf_scale.set(50)
        self.inf_counter_var = tk.StringVar(value="50.0")
        self.inf_counter = customtkinter.CTkLabel(self.frame_params, textvariable=self.inf_counter_var,
                                                  text_font=("Arial", 15),
                                                  width=50)
        self.inf_label.place(x=25, y=150)
        self.inf_scale.place(x=175, y=157)
        self.inf_counter.place(x=330, y=150)

        # Guidance scale
        self.guidance_label = customtkinter.CTkLabel(self.frame_params, text="Guidance scale",
                                                     text_font=("Arial", title_size))
        self.guidance_label.text_label.grid(row=0, column=0)
        self.guidance_var = tk.DoubleVar()
        self.guidance_var.set(12)
        self.guidance_scale = customtkinter.CTkSlider(self.frame_params, variable=self.guidance_var, from_=1, to=20,
                                                      number_of_steps=38,
                                                      width=150, command=self.slider_event_guidance)
        self.guidance_scale.set(12)
        self.guidance_counter_var = tk.StringVar(value="12.0")
        self.guidance_counter = customtkinter.CTkLabel(self.frame_params, textvariable=self.guidance_counter_var,
                                                       text_font=("Arial", 15),
                                                       width=50)
        self.guidance_label.place(x=25, y=180)
        self.guidance_scale.place(x=175, y=187)
        self.guidance_counter.place(x=330, y=180)

        # Seed
        self.seed_label = customtkinter.CTkLabel(self.frame_params, text="Seed", text_font=("Arial", title_size))
        self.seed_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.seed_text = customtkinter.CTkEntry(self.frame_params, width=180)
        self.seed_text.insert(0, "-1")
        self.seed_label.place(x=25, y=210)
        self.seed_text.place(x=175, y=210)

        # Output file
        self.output_label = customtkinter.CTkLabel(self.frame_params, text="Image name",
                                                   text_font=("Arial", title_size))
        self.output_label.text_label.place(relx=0, rely=0.5, anchor="w")
        self.output_text = customtkinter.CTkEntry(self.frame_params, width=180)
        self.output_text.insert(0, "example")
        self.output_label.place(x=425, y=150)
        self.output_text.place(x=585, y=150)

        # Variation 1
        self.varia1_var = customtkinter.StringVar(value="Prompt")
        self.varia1_box = customtkinter.CTkComboBox(self.frame_params, state="readonly",
                                                    values=["Prompt", "Neg prompt", "Inference steps",
                                                            "Guidance scale"],
                                                    variable=self.varia1_var)
        self.varia1_box.entry.configure(readonlybackground="#3d3d3d")
        self.varia1_text = customtkinter.CTkEntry(self.frame_params, width=180)
        self.varia1_box.place(x=425, y=180)
        self.varia1_text.place(x=585, y=180)

        # Variation 2
        self.varia2_var = customtkinter.StringVar(value="Guidance scale")
        self.varia2_box = customtkinter.CTkComboBox(self.frame_params, state="readonly",
                                                    values=["Prompt", "Neg prompt", "Inference steps",
                                                            "Guidance scale"],
                                                    variable=self.varia2_var)
        self.varia2_box.entry.configure(readonlybackground="#3d3d3d")
        self.varia2_text = customtkinter.CTkEntry(self.frame_params, width=180)
        self.varia2_box.place(x=425, y=210)
        self.varia2_text.place(x=585, y=210)

        # Generate
        self.gen_button = customtkinter.CTkButton(parent, text='Generate', command=self.start_thread,
                                                  text_font=("Arial", 30, "bold"),
                                                  width=200, height=70, state="disabled")
        self.gen_button.place(x=840, y=300)

        # Generation speed
        self.speed_var = tk.StringVar()
        self.speed_var.set("")
        self.speed_label = customtkinter.CTkLabel(self, textvariable=self.speed_var, width=100, text_font=("Arial", 10))
        self.speed_label.place(x=900, y=385)

        # Progress bar
        self.prog_bar = customtkinter.CTkProgressBar(master=parent, width=200)
        self.prog_bar.set(0)
        self.prog_bar.place(x=840, y=380)

        # Canvas for image
        self.canvas_out = tk.Canvas(parent, width=512, height=512, bg='white')
        self.canvas_out.place(x=1063, y=69)

        # Seed text box for explore mode
        self.seed1_var = tk.StringVar()
        self.seed2_var = tk.StringVar()
        self.seed3_var = tk.StringVar()
        self.seed4_var = tk.StringVar()
        self.text_seed1 = customtkinter.CTkEntry(parent, width=100, textvariable=self.seed1_var)
        self.text_seed2 = customtkinter.CTkEntry(parent, width=100, textvariable=self.seed2_var)
        self.text_seed3 = customtkinter.CTkEntry(parent, width=100, textvariable=self.seed3_var)
        self.text_seed4 = customtkinter.CTkEntry(parent, width=100, textvariable=self.seed4_var)

        # Trigger the mode button when starting
        self.render_button.invoke()

    def scheduler_choice(self):
        choice = self.scheduler_var.get()
        model = "./stable_diffusion_onnx"
        if choice == "DDPMScheduler":
            self.scheduler = DDPMScheduler.from_pretrained(model, subfolder="scheduler")
        elif choice == "DDIMScheduler":
            self.scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
        elif choice == "PNDMScheduler":
            self.scheduler = PNDMScheduler.from_pretrained(model, subfolder="scheduler")
        elif choice == "LMSDiscreteScheduler":
            self.scheduler = LMSDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
        elif choice == "EulerAncestralDiscreteScheduler":
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
        elif choice == "EulerDiscreteScheduler":
            self.scheduler = EulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
        elif choice == "DPMSolverMultistepScheduler":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler")

        # Load model
        self.pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="DmlExecutionProvider",
                                                                safety_checker=None, scheduler=self.scheduler)

        # Enable generate
        self.gen_button.configure(state="normal")

    def slider_event_guidance(self, value):
        self.guidance_counter_var.set(str(value))

    def slider_event_inference(self, value):
        self.inf_counter_var.set(str(value))

    def mode_changer(self):
        if self.mode_var.get() == "Render":
            try:
                self.text_seed1.place_forget()
                self.text_seed2.place_forget()
                self.text_seed3.place_forget()
                self.text_seed4.place_forget()
            except NameError:
                pass
            self.seed_text.configure(state="normal")
            self.varia1_box.configure(state="disabled")
            self.varia2_box.configure(state="disabled")
            self.varia1_text.configure(state="disabled")
            self.varia2_text.configure(state="disabled")

        elif self.mode_var.get() == "Explore":
            self.text_seed1.place(x=1141, y=30)
            self.text_seed2.place(x=1397, y=30)
            self.text_seed3.place(x=1141, y=590)
            self.text_seed4.place(x=1397, y=590)
            self.varia1_box.configure(state="disabled")
            self.varia2_box.configure(state="disabled")
            self.varia1_text.configure(state="disabled")
            self.varia2_text.configure(state="disabled")
            self.seed_text.configure(state="disabled")

        elif self.mode_var.get() == "Variations":
            try:
                self.text_seed1.place_forget()
                self.text_seed2.place_forget()
                self.text_seed3.place_forget()
                self.text_seed4.place_forget()
            except NameError:
                pass
            self.varia1_box.configure(state="normal")
            self.varia2_box.configure(state="normal")
            self.varia1_text.configure(state="normal")
            self.varia2_text.configure(state="normal")
            self.seed_text.configure(state="normal")

    def start_thread(self):
        # Disable generate buttons
        self.gen_button.configure(state="disabled")
        t = threading.Thread(target=self.generate)
        t.start()

    def stdout_listening(self):
        prog = 0
        while prog < 100:
            output = self.temp_out.getvalue()
            try:
                # Get % of progress from output
                prog_text = re.findall(r"\d{1,3}(?=%)", output)
                prog = int(prog_text[-1])
                self.prog_bar.set(prog / 100)
                it_text = re.findall(r"\d\.\d{1,2}s/it|\d\.\d{1,2}it/s", output)
                it = it_text[-1]
                self.speed_var.set(it)
            except:
                pass
            time.sleep(0.5)

        # Reinitialize speed variable
        self.speed_var.set("")

    def generate(self):
        mode = self.mode_var.get()
        if mode == "Render":
            # Get values
            values = {"Prompt": self.prompt_text.textbox.get("0.0", "end"),
                      "Neg prompt": self.neg_text.textbox.get("0.0", "end"),
                      "Inference steps": self.inf_var.get(),
                      "Guidance scale": self.guidance_var.get(),
                      "Seed": self.seed_text.get(),
                      "Output": f"images/render/{self.output_text.get().replace(' ', '_')}",
                      "temp_name": ""}

            # Start generation script
            generated_seed = self.render(values=values, pipe=self.pipe, width=512, height=512)

            # Display generated image
            self.image = tk.PhotoImage(file=f"{values['Output']}_{generated_seed}.png")
            self.canvas_out.create_image(0, 0, anchor="nw", image=self.image)

        elif mode == "Explore":
            # Get values
            values = {"Prompt": self.prompt_text.textbox.get("0.0", "end"),
                      "Neg prompt": self.neg_text.textbox.get("0.0", "end"),
                      "Inference steps": self.inf_var.get(),
                      "Guidance scale": self.guidance_var.get(),
                      "Output": self.output_text.get().replace(" ", "_")}

            # Start generation script
            seeds = self.explore(values=values, pipe=self.pipe, width=512, height=512)

            # Display generated image and corresponding seeds
            self.image = tk.PhotoImage(file=f"images/explore/{values['Output']}_{seeds[0]}.png")
            self.smaller_image = self.image.subsample(2, 2)
            self.canvas_out.create_image(0, 0, anchor="nw", image=self.smaller_image)

            # Display seeds
            self.seed1_var.set(seeds[0])
            self.seed2_var.set(seeds[1])
            self.seed3_var.set(seeds[2])
            self.seed4_var.set(seeds[3])

        elif mode == "Variations":
            # Recreate temporary sdterr
            self.temp_out = io.StringIO()
            sys.stderr = self.temp_out

            # Get values
            values = {"Prompt": self.prompt_text.textbox.get("0.0", "end"),
                      "Neg prompt": self.neg_text.textbox.get("0.0", "end"),
                      "Inference steps": self.inf_var.get(),
                      "Guidance scale": self.guidance_var.get(),
                      "Seed": self.seed_text.get(),
                      "Output": f"images/variations/{self.output_text.get().replace(' ', '_')}",
                      "temp_name": ""}

            values_varia1 = self.varia1_text.get().split(",")
            param_varia1 = self.varia1_box.get()
            values_varia2 = self.varia2_text.get().split(",")
            param_varia2 = self.varia2_box.get()

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
                    generated_seed = self.render(values=render_values, pipe=self.pipe, width=512, height=512)

            # Merge all generated images into one composite image
            self.merge_variations(images=images_list, varia1=values_varia1, varia2=values_varia2,
                                  name=f"{values['Output']}_{generated_seed}.png")

            # Display generated image
            self.image = tk.PhotoImage(file=f"{values['Output']}_{generated_seed}.png")
            dezoom = len(values_varia1) if len(values_varia1) > len(values_varia2) else len(values_varia2)
            self.smaller_image = self.image.subsample(dezoom + 1, dezoom + 1)
            self.canvas_out.create_image(0, 0, anchor="nw", image=self.smaller_image)

            shutil.rmtree("images/temp")

        # Reactivate generate buttons
        self.gen_button.configure(state="normal")

    def render(self, values, pipe, width, height):
        # Recreate temporary sdterr and start listening
        self.temp_out = io.StringIO()
        sys.stderr = self.temp_out
        l = threading.Thread(target=self.stdout_listening)
        l.start()

        # Generate seed
        if values["Seed"] == "-1":
            rng = np.random.default_rng()
            seed = rng.integers(np.iinfo(np.uint32).max)
        else:
            seed = values["Seed"]

        rng = np.random.RandomState(int(seed))

        image = pipe(values["Prompt"], height, width, int(values["Inference steps"]), float(values["Guidance scale"]),
                     values["Neg prompt"], generator=rng).images[0]

        if values["temp_name"] == "":
            image.save(f"{values['Output']}_{seed}.png")
            with open(f"{values['Output']}_{seed}_parameters.txt", "w") as output:
                output.write(f"{values['Output']} ({width}x{height}) Seed {seed}\n"
                             f"Prompt = {values['Prompt']}\n"
                             f"Negative prompt = {values['Neg prompt']}\n"
                             f"Inference steps = {values['Inference steps']}\n"
                             f"Guidance scale = {values['Guidance scale']}\n")
        else:
            image.save(f"{values['temp_name']}")

        return seed

    def explore(self, values, pipe, width, height):

        if not os.path.exists("images/temp"):
            os.mkdir("images/temp")
        else:
            shutil.rmtree("images/temp")
            os.mkdir("images/temp")

        seeds = []
        composite = Image.new(mode="RGB", size=(1024, 1024))  # Final composition of generated images
        positions = [(0, 0), (512, 0), (0, 512), (512, 512)]  # Anchor positions in the composite image

        # Generate the 4 images with random seeds
        for x in range(4):
            # Recreate temporary sdterr and start listening
            self.temp_out = io.StringIO()
            sys.stderr = self.temp_out
            l = threading.Thread(target=self.stdout_listening)
            l.start()

            # Generate random seed
            rng = np.random.default_rng()
            seed = rng.integers(np.iinfo(np.uint32).max)
            seeds.append(seed)
            rng = np.random.RandomState(int(seed))

            # Generate image
            image = pipe(values["Prompt"], height, width, int(values["Inference steps"]),
                         float(values["Guidance scale"]), values["Neg prompt"], generator=rng).images[0]
            image.save(f"images/temp/image{x}.png")

            # Add generated image
            gen_image = Image.open(f"images/temp/image{x}.png")
            composite.paste(gen_image, positions[x])
            composite.save("images/temp/composite_temp.png")

        # Save final composite image
        composite.save(f"images/explore/{values['Output']}_{seeds[0]}.png")
        shutil.rmtree("images/temp")
        seeds = [str(x) for x in seeds]

        # Save parameters
        with open(f"images/explore/{values['Output']}_{seeds[0]}_parameters.txt", "w") as output:
            output.write(f"{values['Output']} ({width}x{height}) Seeds {' '.join(seeds)}\n"
                         f"Prompt = {values['Prompt']}\n"
                         f"Negative prompt = {values['Neg prompt']}\n"
                         f"Inference steps = {values['Inference steps']}\n"
                         f"Guidance scale = {values['Guidance scale']}\n")

        return seeds

    def merge_variations(self, images, varia1, varia2, name):
        # Create composition image depending on the number of images generated
        size_var1, size_var2 = len(varia1) + 1, len(varia2) + 1  # +1 for title space
        composite = Image.new(mode="RGB", size=(512 * size_var2, 512 * size_var1),
                              color="white")  # Final composition of generated images
        font = ImageFont.truetype(font="arial.ttf", size=65)

        # First column -> names of varia1
        for y in range(1, size_var1):
            position = 512 * y
            text = varia1[y - 1]
            img = Image.new(mode="RGB", size=(512, 512), color="white")
            draw = ImageDraw.Draw(img)
            w, h = draw.textsize(text)
            draw.text((0, (512 - h) / 2), text=text, fill='black', font=font)  # center text
            composite.paste(img, (0, position))

        # First row -> names of varia2
        for x in range(1, size_var2):
            position = 512 * x
            text = varia2[x - 1]
            img = Image.new(mode="RGB", size=(512, 512), color="white")
            draw = ImageDraw.Draw(img)
            w, h = draw.textsize(text)
            draw.text(((512 - w) / 2, (512 - h) / 2), text=text, fill='black', font=font)  # center text
            composite.paste(img, (position, 0))

        # Add the image from left to right (x) and from top to bottom (y)
        counter = 0
        for y in [512 * x for x in range(1, size_var1)]:
            for x in [512 * x for x in range(1, size_var2)]:
                img = Image.open(images[counter])
                composite.paste(img, (x, y))
                counter += 1

        # Save final image
        composite.save(name)


if __name__ == "__main__":
    root = customtkinter.CTk()
    App(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
