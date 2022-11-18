# StableDiffusion-UI

This is a simple UI for stable diffusion using DirectML (https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674) made with tkinter.
With this, you don't need to reload the model for each generation, allowing faster and more pleasant sessions.  
Improvements are still needed but it works well. This is not dummy proof, it might crash of you use inapropriate values (such as letters for the seed).  
More coming (inpainting and image to image interfaces) ...

## Install

After installing Stable diffusion following @averad instructions, simply download the 2 scripts in the same folder.

## Usage

To start the UI, activate the environment then run the gui.py script.
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
