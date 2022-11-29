import shutil
from PIL import Image, ImageDraw, ImageFont
import os
import torch


def render(values, pipe, width, height):
    # Generate seed
    generator = torch.Generator()
    if values["Seed"] == "-1":
        seed = generator.seed()
    else:
        seed = int(values["Seed"])
    generator = generator.manual_seed(seed)
    latents = torch.randn((1, 4, height // 8, width // 8), generator=generator)

    image = pipe(values["Prompt"], height, width, int(values["Inference steps"]), float(values["Guidance scale"]),
                 values["Neg prompt"], float(values["ETA"]), latents=latents,
                 execution_provider="DmlExecutionProvider").images[0]

    if values["temp_name"] == "":
        image.save(f"{values['Output']}_{seed}.png")
        with open(f"{values['Output']}_{seed}_parameters.txt", "w") as output:
            output.write(f"{values['Output']} ({width}x{height}) Seed {seed}\n"
                         f"Prompt = {values['Prompt']}\n"
                         f"Negative prompt = {values['Neg prompt']}\n"
                         f"Inference steps = {values['Inference steps']}\n"
                         f"Guidance scale = {values['Guidance scale']}\n"
                         f"ETA = {values['ETA']}\n")
    else:
        image.save(f"{values['temp_name']}")

    return seed


def explore(values, pipe, width, height):

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
        generator = torch.Generator()
        seed = generator.seed()
        seeds.append(seed)
        generator = generator.manual_seed(seed)
        latents = torch.randn((1, 4, height // 8, width // 8), generator=generator)

        image = pipe(values["Prompt"], height, width, int(values["Inference steps"]), float(values["Guidance scale"]),
                     values["Neg prompt"], float(values["ETA"]), latents=latents,
                     execution_provider="DmlExecutionProvider").images[0]
        image.save(f"images/temp/image{x}.png")
        image = None

        # Add generated image
        gen_image = Image.open(f"images/temp/image{x}.png")
        composite.paste(gen_image, positions[x])

    composite.save(f"images/explore/{values['Output']}_{seeds[0]}.png")
    shutil.rmtree("images/temp")
    seeds = [str(x) for x in seeds]

    with open(f"images/explore/{values['Output']}_{seeds[0]}_parameters.txt", "w") as output:
        output.write(f"{values['Output']} ({width}x{height}) Seeds {' '.join(seeds)}\n"
                     f"Prompt = {values['Prompt']}\n"
                     f"Negative prompt = {values['Neg prompt']}\n"
                     f"Inference steps = {values['Inference steps']}\n"
                     f"Guidance scale = {values['Guidance scale']}\n"
                     f"ETA = {values['ETA']}\n")

    return seeds


def merge_variations(images, varia1, varia2, name):
    # Create composition image depending on the number of images generated
    size_var1, size_var2 = len(varia1) + 1, len(varia2) + 1  # +1 for title space
    composite = Image.new(mode="RGB", size=(512 * size_var2, 512 * size_var1), color="white")  # Final composition of generated images
    font = ImageFont.truetype(font="arial.ttf", size=65)

    # First column -> names of varia1
    for y in range(1, size_var1):
        position = 512*y
        text = varia1[y-1]
        img = Image.new(mode="RGB", size=(512, 512), color="white")
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(text)
        draw.text((0, (512 - h)/2), text=text, fill='black', font=font)  # center text
        composite.paste(img, (0, position))

    # First row -> names of varia2
    for x in range(1, size_var2):
        position = 512*x
        text = varia2[x-1]
        img = Image.new(mode="RGB", size=(512, 512), color="white")
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(text)
        draw.text(((512-w)/2, (512-h)/2), text=text, fill='black', font=font)   # center text
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


def inpainting(values, pipe):
    image = pipe(prompt=values["Prompt"], image=Image.open(values["Image"]), mask_image=Image.open("images/mask.png"),
                 guidance_scale=values["Guidance scale"]).images[0]
    image.save(f"{values['Output']}.png")
