import os
import time
import requests
import argparse
import urllib.parse
import concurrent.futures
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, unquote
import json
import random
import hashlib
import io
from PIL import Image
import re

def setup_driver(browser='firefox', headless=True):
    """Set up the web driver for browser automation."""
    if browser.lower() == 'firefox':
        firefox_options = FirefoxOptions()
        if headless:
            firefox_options.add_argument("--headless")
        firefox_options.add_argument("--window-size=1920,1080")
        firefox_options.set_preference("browser.download.folderList", 2)
        firefox_options.set_preference("browser.download.manager.showWhenStarting", False)
        return webdriver.Firefox(
            service=FirefoxService(GeckoDriverManager().install()),
            options=firefox_options
        )
    else:  # Chrome
        chrome_options = ChromeOptions()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        return webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options
        )

def get_image_urls_from_getty(person_name, num_images=60, browser='firefox'):
    """Get image URLs from Getty Images search."""
    # Create a query string for Getty Images
    query = urllib.parse.quote(f"{person_name} portrait")
    search_url = f"https://www.gettyimages.fr/search/2/image?family=editorial&phrase={query}"
    
    # Set up the driver
    driver = setup_driver(browser)
    
    try:
        # Navigate to the search URL
        driver.get(search_url)
        
        # Wait for images to load
        time.sleep(3)
        
        # Scroll down to load more images
        for _ in range(5):  # Adjust this value to load more or fewer images
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for the page to load
        
        # Get image elements
        image_elements = driver.find_elements(By.CSS_SELECTOR, "img.Xc8V0Fvh0qg0lUySLpoi")
        
        # Extract image URLs
        image_urls = []
        for img in image_elements:
            if img.get_attribute("src") and img.get_attribute("src").startswith("http"):
                image_urls.append(img.get_attribute("src"))
        
        # Deduplicate
        image_urls = list(set(image_urls))
        
        print(f"Found {len(image_urls)} images for {person_name} on Getty Images")
        return image_urls[:min(num_images, len(image_urls))]
    
    finally:
        driver.quit()

def download_image(url, output_dir, person_name, index):
    """Download an image from a URL and save it to the output directory."""
    try:
        response = requests.get(url, timeout=10)
        
        # Check if the response was successful
        if response.status_code == 200:
            # Generate a filename
            file_extension = 'jpg'  # Default extension for Getty Images
            
            # Create a unique filename
            filename = f"{person_name.replace(' ', '_')}_{index}.{file_extension}"
            file_path = os.path.join(output_dir, filename)
            
            # Save the image
            try:
                # Check if it's a valid image by opening it with PIL
                img = Image.open(io.BytesIO(response.content))
                
                # Convert to RGB if necessary (e.g., if it's RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Check image size
                width, height = img.size
                if width < 100 or height < 100:
                    return False, f"Image too small: {width}x{height}"
                
                # Save the image
                img.save(file_path)
                return True, file_path
            
            except Exception as e:
                return False, f"Error processing image: {str(e)}"
        else:
            return False, f"Failed to download image: HTTP {response.status_code}"
    
    except Exception as e:
        return False, f"Error downloading image: {str(e)}"

def collect_person_images(person_name, output_dir, num_images=30, browser='firefox'):
    """Collect images for a specific person."""
    # Create a directory for the person
    person_dir = os.path.join(output_dir, person_name.replace(' ', '_'))
    os.makedirs(person_dir, exist_ok=True)
    
    # Get image URLs from Getty Images
    print(f"Searching for images of {person_name}...")
    image_urls = get_image_urls_from_getty(person_name, num_images=num_images, browser=browser)
    
    # Download images
    print(f"Found {len(image_urls)} images. Downloading...")
    successful_downloads = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_image, url, person_dir, person_name, i) 
                  for i, url in enumerate(image_urls)]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            success, result = future.result()
            if success:
                successful_downloads += 1
    
    print(f"Successfully downloaded {successful_downloads} images for {person_name}")
    return successful_downloads

def main():
    parser = argparse.ArgumentParser(description="Collect facial images for training from Getty Images")
    parser.add_argument("--output", type=str, default="data/raw", 
                        help="Output directory for the dataset")
    parser.add_argument("--browser", type=str, default="firefox", choices=["firefox", "chrome"],
                        help="Browser to use for scraping")
    parser.add_argument("--people", type=str, nargs="+", 
                        help="List of people to collect images for")
    parser.add_argument("--file", type=str, 
                        help="Text file with list of people (one per line)")
    parser.add_argument("--num-images", type=int, default=60,
                        help="Number of images to download per person")
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Get the list of people
    people = []
    if args.people:
        people.extend(args.people)
    
    if args.file:
        with open(args.file, 'r') as f:
            people.extend([line.strip() for line in f.readlines() if line.strip()])
    
    # If no people were specified, prompt the user
    if not people:
        print("Please enter the names of people to collect images for (one per line).")
        print("Enter an empty line when you're done.")
        
        while True:
            name = input("> ").strip()
            if not name:
                break
            people.append(name)
    
    # Collect images for each person
    total_images = 0
    for person in people:
        images_downloaded = collect_person_images(
            person, args.output, num_images=args.num_images, browser=args.browser
        )
        total_images += images_downloaded
    
    print(f"Dataset collection complete. Downloaded a total of {total_images} images.")

if __name__ == "__main__":
    main()