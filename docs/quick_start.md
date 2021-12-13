# Quick Start

## Create a Python Virtual Environment

Although not a necessary step, creating a virtual enviroment is always recommended for libraries that tend to install more packages ontop of themselves. Nobody likes a cluttered python install, so we will quickly go over the steps.

1. Decide on a location to install the virtual enviroment. This location will contain all the files for the virtual enviroment and the venv will be activated from this location. Here are the commands for a POSIX and Windows system: 
   1. `python3 -m venv /path/to/venv`
   2. `c:\>c:\Python35\python -m venv c:\path\to\venv`
   
2. Start the virtual environment using the command corresponds to your system:

| Platform | Shell | Command to activate virtual environment |
|---|---|---|
| Posix | bash/zsh | $ source <path/to/venv>/bin/activate |
| | fish | $ source <path/to/venv>/bin/activate.fish |
| | csh/tcsh | $ source <path/to/venv>/bin/activate.csh |
| | PowerShell Core | $ <path/to/venv>/bin/Activate.ps1 |
| Windows | cmd.exe | C:\> <path\to\venv>\Scripts\activate.bat |
| | PowerShell | PS C:\> <path\to\venv>\Scripts\Activate.ps1 |

## Install Towhee

Towhee can be installed in one of two ways, through github and through pip. Lets begin with the easiest route, the pip route.

1. Pip
   1. First, if using a venv, activate it before moving ahead.
   2. Next, install towhee using the following line: pip install towhee
   3. That's it!
2. Github
   1. First, if using a venv, activate it before moving ahead.
   2. If not already present, install git on your system.
   3. Next, clone the towhee repository:
   `git clone https://github.com/towhee-io/towhee.git`
   4. After the download is done, proceed into the towhee directory and run the following command:
   `python setup.py install`

## Run Image Pipeline

For this quick start we are going to use the easy image embedding pipeline. Once this code is run, the needed files will be downloaded from hub.towhee.io and the embedding results will be given in the output variable.

```
# Import pipeline from towhee, pipeline is the main function in towhee. 
from towhee import pipeline
# Import pillow to be able to open images.
from PIL import Image

# Begin by opening the image, in this example we are using a local file on
# our end, replace the file with the image you want to embed. Since this
# pipeline uses the RGB format, convert the image to RGB aswell.
img = Image.open('towhee_logo.png').convert('RGB')
# Initiate the pipeline.
embedding_pipeline = pipeline('image-embedding')
# Run image through pipeline. 
output = embedding_pipeline(img)
# Print the output image.
print(output)
```

## Conclusion

And that is it, with this guide you were able to install towhee and run your first pipeline. In the future we will be releasing more guides to help you make your own pipelines and run other peoples pipelines. Stay tuned. 

