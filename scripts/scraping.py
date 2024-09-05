import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# Function to download images
def download_images(img_urls, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, url in enumerate(img_urls):
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))

            # Convert RGBA images to RGB before saving as JPEG
            if image.mode in ("RGBA", "P"):  # Check if the image has an alpha channel
                image = image.convert("RGB")  # Convert to RGB
            
            # Determine file extension based on image mode
            if image.mode == "RGB":
                image_path = os.path.join(output_dir, f"non_menstrual_{i+1}.jpg")
                image.save(image_path, "JPEG")  # Save as JPEG
            else:
                # Save as PNG if image mode is not RGB
                image_path = os.path.join(output_dir, f"non_menstrual_{i+1}.png")
                image.save(image_path, "PNG")  # Save as PNG
            
            print(f"Downloaded {image_path}")
        except Exception as e:
            print(f"Could not download {url} - {e}")

# Function to scrape images using BeautifulSoup
def scrape_images(search_url, output_dir):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    img_tags = soup.find_all("img")
    
    img_urls = []
    for img in img_tags:
        img_url = img.get("src")
        if img_url and img_url.startswith("http"):
            img_urls.append(img_url)

    # Download the images
    download_images(img_urls, output_dir)

# URL of the website you want to scrape images from
search_url = "https://www.thoughtco.com/municipal-waste-and-landfills-overview-1434949"  # Replace this with a relevant website for menstrual waste

# Set output directory to your specified path
output_dir = r"C:\Users\TATHAGATA GHOSH\Desktop\hackX_env\Dataset\Tilottoma\dataset"

# Scrape the images and save them
scrape_images(search_url, output_dir)
