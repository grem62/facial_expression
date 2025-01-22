import os
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
from PIL import Image
import imagehash
from tqdm import tqdm

def clean_images(input_folder, output_folder, target_size=(224, 224), remove_duplicates=True):
    """
    Cleans, resizes, and prepares images for machine learning.
    
    Args:
        input_folder (str): Folder containing raw images.
        output_folder (str): Folder to save cleaned images.
        target_size (tuple): Size of the output images (width, height).
        remove_duplicates (bool): Remove duplicate images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Dictionary to store image hashes
    image_hashes = set()
    
    for filename in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image
        if not filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            continue
        
        try:
            # Read the image
            img = cv2.imread(input_path)
            
            # Convert to grayscale (optional, depending on the model)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Denoise the image
            img = denoise_wavelet(img, channel_axis=-1, rescale_sigma=True)

            # Resize the image
            img = cv2.resize((img * 255).astype(np.uint8), target_size, interpolation=cv2.INTER_AREA)
            
            # Check for duplicates
            if remove_duplicates:
                pil_img = Image.fromarray(img)
                img_hash = str(imagehash.average_hash(pil_img))
                
                if img_hash in image_hashes:
                    continue  # Skip duplicates
                image_hashes.add(img_hash)
            
            # Save the cleaned image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)
        
        except Exception as e:
            print(f"Error with image {filename}: {e}")

# Example usage
input_folder = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/raw/data_set_comprehension_image/train/surprise"
output_folder = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/raw/dataset_fusion/train/surprise"
clean_images(input_folder, output_folder)
