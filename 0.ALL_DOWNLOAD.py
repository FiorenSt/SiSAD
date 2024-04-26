import requests
from datetime import datetime, timedelta
import tarfile
import argparse
import random
import fastavro
import h5py
import numpy as np
import os
import stat


def generate_date_urls(start_date, end_date, url_template, seed=42):
    """
    Generates URLs based on a range of dates and shuffles them to randomize the order.
    Uses a seed for reproducible shuffling.

    :param start_date: Start date in "YYYYMMDD" format
    :param end_date: End date in "YYYYMMDD" format
    :param url_template: URL template with a placeholder for the date
    :param seed: Seed for the random number generator to ensure reproducibility.
    :return: List of URLs with dates, in randomized order
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    date_generated = [start + timedelta(days=x) for x in range(0, (end-start).days + 1)]

    urls = [url_template.format(date=date.strftime("%Y%m%d")) for date in date_generated]
    random.seed(seed)  # Set the seed for reproducible shuffling
    random.shuffle(urls)
    return urls



def safe_extract_tarfile(filepath, extract_to):
    """
    Safely extracts files from a tar.gz archive, skipping files that cause a PermissionError.

    :param filepath: Path to the tar.gz file.
    :param extract_to: Directory where contents will be extracted.
    """
    with tarfile.open(filepath, "r:gz") as tar:
        for member in tar.getmembers():
            try:
                tar.extract(member, path=extract_to)
            except PermissionError as e:
                print(f"Permission error occurred while extracting {member.name}: {e}. Skipping this file.")
            except Exception as e:
                print(f"An unexpected error occurred while extracting {member.name}: {e}. Skipping this file.")


def download_and_unzip(url, extract_to, min_file_size):
    """
    Download and extract a single tar.gz file from the given URL.
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"Downloaded {filename}")

        if os.path.getsize(filepath) > min_file_size and filename.endswith("tar.gz"):
            safe_extract_tarfile(filepath, extract_to)
            print(f"Extracted {filename} in {extract_to}")
            os.remove(filepath)
            print(f"Removed downloaded file: {filename}")
        else:
            print(f"File {filename} was too small or not a tar.gz file and has been removed.")
            os.remove(filepath)
            return False  # Indicates failed extraction due to file size
    except Exception as e:
        print(f"An unexpected error occurred while processing {filename}: {e}. Skipping this file.")
        os.remove(filepath)
        return False  # Indicates failed extraction due to error
    return True  # Indicates successful extraction


def shuffle_avro_file_paths(folder_path, seed=42):
    """
    Collects .avro file paths in the given folder and shuffles them.

    :param folder_path: The folder containing .avro files.
    :param seed: Seed for the random number generator to ensure reproducibility.
    :return: A shuffled list of .avro file paths.
    """
    random.seed(seed)
    file_paths = list(Path(folder_path).glob('*.avro'))
    random.shuffle(file_paths)
    return file_paths

def read_avro_files(file_paths):
    """
    Generator to read records from a list of shuffled .avro file paths.

    :param file_paths: A list of shuffled .avro file paths.
    """
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            reader = fastavro.reader(f)
            for record in reader:
                yield record

# Usage example
folder_path = 'D:/STAMP_AD_IMAGES/Data'  # Adjust as needed
shuffled_file_paths = shuffle_avro_file_paths(folder_path)
records = read_avro_files(shuffled_file_paths)



def extract_fits_image(fits_bytes):
    """
    Extracts FITS image data from bytes, converts to float64, replaces NaNs with 0.
    """
    import gzip
    from astropy.io import fits
    import io
    with gzip.open(io.BytesIO(fits_bytes), 'rb') as gz:
        with fits.open(io.BytesIO(gz.read())) as hdul:
            image_data = hdul[0].data.astype(np.float64)
            image_data = np.nan_to_num(image_data)
            return image_data

from pathlib import Path

def get_next_file_number(output_folder):
    """
    Determines the next file number to use based on existing files in the output folder.
    Assumes files are named in the format 'data_X.h5' where X is an integer.
    """
    existing_files = list(Path(output_folder).glob('data_*.h5'))
    if not existing_files:
        return 0
    else:
        # Extract numbers from all file names and find the maximum
        max_number = max(int(file.stem.split('_')[-1]) for file in existing_files)
        return max_number + 1  # Next file number

def save_triplets_and_features_in_batches(records, output_folder, batch_size=1024):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    batch_images = []
    batch_objectIds = []
    batch_candids = []
    batch_other_features = []
    file_number = get_next_file_number(output_folder)  # Get the next file number

    for record in records:
        triplet_images = []
        correct_size = True
        for image_type in ['Science', 'Template', 'Difference']:
            fits_bytes = record[f'cutout{image_type}']['stampData']
            image = extract_fits_image(fits_bytes)
            if image.shape == (63, 63):
                triplet_images.append(image)
            else:
                correct_size = False
                break

        if correct_size:
            candidate = record.get('candidate', {})
            objectId = record.get('objectId', 'NoObjectId')  # Correctly access 'objectId' from the record
            candid = int(candidate.get('candid', 0))  # Stored separately

            # Numeric features extracted here
            numeric_features = [
                candidate.get('rb', np.nan),  # Real bogus score
                candidate.get('drb', np.nan),  # Deep-learning real bogus score
                candidate.get('fid', np.nan),  # Filter ID
                candidate.get('sciinpseeing', np.nan),  # Seeing science image
                candidate.get('magpsf', np.nan),  # Magnitude from PSF-fit
                candidate.get('sigmapsf', np.nan),  # 1-sigma uncertainty in magpsf
                candidate.get('classtar', np.nan),  # Star/Galaxy classification score
                candidate.get('ra', np.nan),  # Right Ascension
                candidate.get('dec', np.nan),  # Declination
                candidate.get('fwhm', np.nan),  # Full Width Half Max
                candidate.get('aimage', np.nan),  # Major axis
                candidate.get('bimage', np.nan),  # Minor axis
                candidate.get('elong', np.nan),  # Elongation
                candidate.get('nbad', np.nan),  # Number of bad pixels
                candidate.get('nneg', np.nan),  # Number of negative pixels
                candidate.get('jd', np.nan),  # Julian date
                candidate.get('ndethist', np.nan),  # Detection history metrics
                candidate.get('ncovhist', np.nan),
                candidate.get('jdstarthist', np.nan),
                candidate.get('jdendhist', np.nan),
            ]

            batch_images.append(np.stack(triplet_images, axis=0))
            batch_objectIds.append(objectId)
            batch_candids.append(candid)
            batch_other_features.append(numeric_features)

            if len(batch_images) == batch_size:
                save_batch(output_folder, file_number, batch_images, batch_objectIds, batch_candids, batch_other_features)
                # Resetting the batches
                batch_images, batch_objectIds, batch_candids, batch_other_features = [], [], [], []
                file_number += 1



def safe_remove(file_path):
    try:
        # Remove read-only attribute, if it's set
        os.chmod(file_path, stat.S_IWRITE)

        # Attempt to delete the file
        os.remove(file_path)
        # print(f"Successfully removed {file_path}")
    except PermissionError as e:
        print(f"PermissionError: Could not remove {file_path}. Error: {e}")
    except Exception as e:
        print(f"Error: Could not remove {file_path}. Error: {e}")


def save_batch(output_folder, file_number, batch_images, batch_objectIds, batch_candids, batch_other_features):
    with h5py.File(f"{output_folder}/data_{file_number}.h5", 'w') as hf:
        hf.create_dataset('images', data=np.array(batch_images))
        # Specify UTF-8 encoding for objectIds
        objectId_dtype = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('objectIds', data=np.array(batch_objectIds, dtype=object), dtype=objectId_dtype)
        hf.create_dataset('candids', data=np.array(batch_candids, dtype=np.int64))
        hf.create_dataset('features', data=np.array(batch_other_features, dtype=np.float64))
    print(f"Saved batch in data_{file_number}.h5")


def process_avro_data(folder_path, output_folder, batch_size=1024):
    shuffled_file_paths = shuffle_avro_file_paths(folder_path)
    records = read_avro_files(shuffled_file_paths)
    print('Processing Records...')
    save_triplets_and_features_in_batches(records, output_folder, batch_size)
    print('Done Saving Triplets.')


def process_and_cleanup_avro_batch(folder_path, output_folder, batch_size = 1024, min_batch_size=1024*100):
    """
    Processes a batch of AVRO files into H5 and then cleans up the AVRO files.
    """
    avro_file_paths = shuffle_avro_file_paths(folder_path)
    if len(avro_file_paths) >= min_batch_size:
        records = read_avro_files(avro_file_paths[:min_batch_size])
        print('Processing Records...')
        save_triplets_and_features_in_batches(records, output_folder, batch_size)
        print('Done Saving Triplets. Cleaning up processed AVRO files...')
        for file_path in avro_file_paths[:min_batch_size]:
            safe_remove(file_path)  # Use safe_remove instead of os.remove
        print('Cleanup completed.')



def main(start_date, end_date, extract_to, output_folder, min_file_size, batch_size, min_batch_size):
    url_template = 'https://ztf.uw.edu/alerts/public/ztf_public_{date}.tar.gz'
    urls = generate_date_urls(start_date, end_date, url_template)

    for url in urls:
        success = download_and_unzip(url, extract_to, min_file_size)
        if not success:
            continue  # Skip to the next URL if extraction failed

        avro_files = list(Path(extract_to).glob('*.avro'))
        if len(avro_files) >= min_batch_size:
            process_and_cleanup_avro_batch(extract_to, output_folder, batch_size, min_batch_size)

        # Implement exit condition if necessary (e.g., based on user input or all URLs processed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download, process, and manage AVRO files efficiently.")
    parser.add_argument('--start_date', required=True, help="Start date in 'YYYYMMDD' format")
    parser.add_argument('--end_date', required=True, help="End date in 'YYYYMMDD' format")
    parser.add_argument('--extract_to', required=True, help="Directory where tar.gz contents will be extracted")
    parser.add_argument('--output_folder', required=True, help="Directory where processed data will be saved")
    parser.add_argument('--min_file_size', type=int, default=512, help="Minimum file size in bytes to proceed with extraction")
    parser.add_argument('--batch_size', type=int, default=1024, help="Number of AVRO files in each h5 file")
    parser.add_argument('--min_batch_size', type=int, default=1024*100, help="Minimum number of AVRO files to trigger processing")

    args = parser.parse_args()

    main(args.start_date, args.end_date, args.extract_to, args.output_folder, args.min_file_size, args.batch_size, args.min_batch_size)



# EXAMPLE CODE
# python 0.ALL_DOWNLOAD.py --start_date 20240101 --end_date 20240421 --extract_to "D:/STAMP_AD_IMAGES/Data" --output_folder "D:/STAMP_AD_IMAGES/ProcessedData" --min_file_size 512 --batch_size 1024 --min_batch_size 102400