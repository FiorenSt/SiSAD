import requests
from bs4 import BeautifulSoup
import os
import random

# Constants
BASE_URL = "https://ztf.uw.edu/alerts/public/"
BATCH_SIZE_GB = 30
BYTE_TO_GB = 1024**3

def get_file_links_and_sizes(base_url):
    """
    Scrapes the file links and their sizes from the given URL.
    
    :param base_url: The base URL to scrape
    :return: A list of tuples containing file links and their sizes in bytes
    """
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = []
    
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.tar.gz'):
            file_url = os.path.join(base_url, href)
            response = requests.head(file_url)
            file_size = int(response.headers.get('content-length', 0))
            links.append((file_url, file_size))
    
    return links

def divide_into_batches(links, batch_size_gb):
    """
    Divides the file links into batches where the sum of the sizes of the files
    in each batch is around the specified batch size in GB.
    
    :param links: A list of tuples containing file links and their sizes in bytes
    :param batch_size_gb: Desired batch size in GB
    :return: A list of batches where each batch is a list of file links
    """
    # Shuffle the links before dividing into batches
    random.shuffle(links)
    
    batches = []
    current_batch = []
    current_batch_size = 0
    max_batch_size = batch_size_gb * BYTE_TO_GB
    
    for link, size in links:
        if current_batch_size + size > max_batch_size:
            batches.append(current_batch)
            current_batch = []
            current_batch_size = 0
        
        current_batch.append(link)
        current_batch_size += size
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def save_batches_to_files(batches, output_dir):
    """
    Saves the batches to individual text files.
    
    :param batches: A list of batches where each batch is a list of file links
    :param output_dir: Directory where the batch files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, batch in enumerate(batches):
        with open(os.path.join(output_dir, f'batch_{i + 1}.txt'), 'w') as f:
            for link in batch:
                f.write(f"{link}\n")

def main():
    links = get_file_links_and_sizes(BASE_URL)
    batches = divide_into_batches(links, BATCH_SIZE_GB)
    save_batches_to_files(batches, 'batches')

if __name__ == '__main__':
    main()
