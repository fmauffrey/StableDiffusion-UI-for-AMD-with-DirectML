# StableDiffusion-UI

This is a simple UI for stable diffusion using DirectML (https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674) made with tkinter.
With this, you don't need to reload the model for each generation, allowing faster and more pleasant sessions.  
Improvements are still needed but it works well. This is not dummy proof, it might crash of you use inapropriate values (such as letters for the seed).  

Ugly look for the moment. Better to come.

## Install

After installing Stable diffusion following @averad instructions, simply download the 3 scripts in the same folder.

## Dependencies
```
pip install customtkinter
```

## Main GUI

To start the GUI, activate the environment then run the gui.py script.
```
sd_env\scripts\activate
python gui.py
```

All images are saved in the images folder and in the corresponding subfolder. All images are generated with a size of 512*512. The image canvas of the ui display images at this size and might not be very appropriate for variations mode. Don't hesitate to look directly at the image file in the variations subfolder for a better view. 

### Render
This mode aims at rendering the best image once you found an appropriate seed, prompt and parameters. You can also just use it to generate any image.

### Explore
Use the explore mode to quickly find the appropriate seed for a specific prompt. This mode will generate 4 images with different and random seeds and displays all the images with the corresponding seeds.
You should use a low number of inference steps in this mode to quickly generate the 4 images and then use the best seed in render or variations mode.

### Variations
This mode allows to generate the same image with some variations and plot them in a grid. Use it after finding a good seed. Do not use random seed (-1) in this mode.
For each category, separate the different values by a comma "," with no spaces.
For the prompt and negative prompt category, use $REPLACE in your text to designate the term to replace.

## Inpainting GUI

To start the GUI, activate the environment then run the gui_inpainting.py script.
```
sd_env\scripts\activate
python gui_inpainting.py
```
Here you will have 2 picture frames, for input (left) and output (right). Load the input image using the "load" button. The image will be automatically resized to 512*512 and appear in the left image frame. Directly paint in white the areas you want to regenerate by clicking on the canvas. You can modify the brush size and clear the image by clicking on the "clear" button. Everything not in pure white will be converted into black. You can also paint in black on the canvas but use it only to cover white areas on your image that you do not want to be regenerated. After setting your prompts and parameters, click on "Generate" to start the inpainting. The output image will appear in the right frame and saved in "/images/inpainting".
