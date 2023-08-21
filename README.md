
Colorizing Images using Pre-trained Models
This is a Python script for colorizing grayscale images using pre-trained models. The script loads two pre-trained models - ECCV 2016 and SIGGRAPH 2017 - to generate colorized versions of input grayscale images. The script also allows for the use of GPU processing if specified in the command line arguments.

Usage
To use this script, follow these steps:

Clone this repository and navigate to the project directory.

Install the required dependencies using pip install -r requirements.txt.

Place the grayscale input image in the imgs folder.

Open a terminal window in the project directory and run the following command:

css
Copy code
python colorize.py -i <input_image> -o <output_file_prefix> --use_gpu
where <input_image> is the name of the grayscale input image in the imgs folder (including the file extension), <output_file_prefix> is the prefix for the saved output file, and the --use_gpu flag is optional and specifies whether to use GPU processing.

For example, to colorize an image named test.jpg with GPU processing and save the output files with the prefix result, run the following command:

css
Copy code
python colorize.py -i test.jpg -o result --use_gpu
The colorized output images will be saved in the project directory with the prefix specified in the command line argument. The original, input, and output images will also be displayed for comparison.
