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
            image_path = os.path.join(output_dir, f"menstrual_waste_image_{i+1}.jpg")
            image.save(image_path)
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
search_url = "https://www.bing.com/images/search?q=used+sanitary+pad&qs=MM&form=QBILPG&sp=2&lq=0&pq=used+sanit&sc=10-10&sk=MM1&cvid=5847FE6772564CEC8862B3213CE0E897&ghsh=0&ghacc=0&first=1"  # Replace this with a relevant website for menstrual waste

# Set output directory to your specified path
output_dir = r"C:\Users\SBasu\.conda\envs\Tilottoma\database"

# Scrape the images and save them
scrape_images(search_url, output_dir)
