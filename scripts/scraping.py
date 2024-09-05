import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
from urllib.parse import urljoin
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import random

# Base URL of the page to scrape
url = 'https://www.pexels.com/search/waste/'

# Function to create a random delay
def random_delay():
    delay = random.uniform(1, 5)
    print(f"Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)

# Setup Selenium with headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

# Use Selenium to get the page source
print(f"Attempting to connect to {url} using Selenium...")
try:
    driver.get(url)
    random_delay()  # Add a random delay
    html = driver.page_source
    print("Successfully retrieved the HTML content using Selenium.")
except Exception as e:
    print(f"Failed to retrieve the content with Selenium. Error: {e}")
    driver.quit()
    raise SystemExit(e)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
print("HTML content parsed successfully.")
driver.quit()

# Directory to save images
category = 'household_waste'
base_dir = os.path.join(r"C:\Users\TATHAGATA GHOSH\Desktop\hackX_env\Dataset\Tilottoma\dataset\non_menstrual", category)
os.makedirs(base_dir, exist_ok=True)
print(f"Images will be saved in: {base_dir}")

# Function to download images with debugging info
def download_images(img_tags):
    ua = UserAgent()
    for i, img in enumerate(tqdm(img_tags[:10], desc="Downloading images", unit="image")):
        try:
            # Use fake_useragent to get a random User-Agent
            headers = {'User-Agent': ua.random}

            # Check for <picture> element to prioritize webp images
            picture = img.find_parent('picture')
            if picture:
                # Prioritize webp source if available
                source_tag = picture.find('source', type='image/webp')
                if source_tag and 'srcset' in source_tag.attrs:
                    img_url = source_tag['srcset']
                else:
                    # Fallback to img src if no webp source is found
                    img_url = img['src']
            else:
                # Handle regular img tags
                img_url = img['src']

            # Convert relative URLs to absolute URLs
            img_url = urljoin(url, img_url)
            print(f"Fetching image URL: {img_url}")

            # Ensure all images are saved in jpg, png, or jpeg format
            img_extension = img_url.split('.')[-1].split('?')[0].lower()
            if img_extension not in ['jpg', 'jpeg', 'png']:
                img_extension = 'jpg'  # Default to jpg if another format is encountered

            img_name = os.path.join(base_dir, f"household_waste_{i + 1}.{img_extension}")
            print(f"Downloading and saving image as: {img_name}")

            # Attempt to download the image
            img_data = requests.get(img_url, headers=headers)
            img_data.raise_for_status()  # Ensure the image was fetched successfully
            with open(img_name, 'wb') as handler:
                handler.write(img_data.content)
            print(f"Successfully downloaded {img_name}")

            random_delay()  # Random delay to simulate human-like behavior

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {img_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# Find all img tags and filter by valid extensions
print("Searching for images on the page...")
img_tags = soup.find_all('img')

# Filter to ensure only images with valid extensions are processed
filtered_img_tags = [img for img in img_tags if img.get('src') and img['src'].split('.')[-1].split('?')[0].lower() in ['jpg', 'jpeg', 'png']]

print(f"Found {len(filtered_img_tags)} valid images on the page.")

if filtered_img_tags:
    download_images(filtered_img_tags)
else:
    print("No valid images found on the page.")
