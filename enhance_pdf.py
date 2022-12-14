import os
import sys

dll_folder_path = os.path.join(os.path.dirname(__file__), "dlls")
sys.path.append(dll_folder_path)
import cv2
import fitz
import numpy as np
import tqdm
from PIL import Image, ImageFile

from ncnnCugan import RealCUGAN
from ncnnRealESR import RealESRGAN
from ncnnWaifu2x import Waifu2x

ImageFile.LOAD_TRUNCATED_IMAGES = True
pdf_file = sys.argv[1]

enhancer = RealESRGAN(model="realesrgan-x4plus")
# enhancer = RealCUGAN(model="up3x-conservative")
# enhancer = Waifu2x(model="cunet-scale2x_model")

img_list = []
img_cv2_list = []
enhanced_list = []

# Load pdf file
pdf_doc = fitz.open(pdf_file)
print(f"Loading {pdf_file}...")
for page in tqdm.tqdm(pdf_doc):
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_list.append(img)
    img_cv2_list.append(img_cv2)
print("- {} images".format(len(img_list)))

for img in tqdm.tqdm(img_cv2_list):
    enhanced_cv2 = enhancer.enhance(img)
    enhanced = Image.fromarray(cv2.cvtColor(enhanced_cv2, cv2.COLOR_BGR2RGB))
    enhanced_list.append(enhanced)

# Save enhanced pdf file
to_file = os.path.splitext(pdf_file)[0] + "_enhanced.pdf"
if os.path.exists(to_file):
    os.remove(to_file)
enhanced_list[0].save(to_file, "pdf", save_all=True, append_images=enhanced_list[1:])
print("Enhanced pdf file saved to {}".format(to_file))
