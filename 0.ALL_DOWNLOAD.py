import requests
from datetime import datetime, timedelta
import tarfile
import argparse
import random
import fastavro
import numpy as np
import os
import stat
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


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


def extract_fits_image(fits_bytes):
    """
    Extracts FITS image data from bytes, converts to float64, replaces NaNs with 0.
    """
    import gzip
    from astropy.io import fits
    import io
    with gzip.open(io.BytesIO(fits_bytes), 'rb') as gz:
        with fits.open(io.BytesIO(gz.read())) as hdul:
            image_data = hdul[0].data.astype(np.float32)
            image_data = np.nan_to_num(image_data)
            return image_data


def get_next_file_number(output_folder):
    """
    Determines the next file number to use based on existing files in the output folder.
    Assumes files are named in the format 'data_X.parquet' where X is an integer.
    """
    existing_files = list(Path(output_folder).glob('data_*.parquet'))
    if not existing_files:
        return 0
    else:
        # Extract numbers from all file names and find the maximum
        max_number = max(int(file.stem.split('_')[-1]) for file in existing_files)
        return max_number + 1  # Next file number


IMAGE_SIZE = 63
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
TRIPLET_SHAPE = (3,) + IMAGE_SHAPE

PARQUET_SCHEMA = pa.schema([
    # pa.fixed_shape_tensor causes an error:
    # *** pyarrow.lib.ArrowNotImplementedError: extension
    # Probably because it is an extension type and it doesn't have a good support yet
    # ('triplet', pa.fixed_shape_tensor(pa.float32(), TRIPLET_SHAPE)),  # image triplet data
    ('triplet', pa.list_(pa.float32(), np.prod(TRIPLET_SHAPE))),  # alert stamps data
    ('objectId', pa.string()),                                    # ZTF Alert object ID, like ZTF18abcxyz
    ('candid', pa.uint64()),                                      # ZTF Alert candidate ID
    ('rb', pa.float32()),                                         # Real-Bogus score
    ('drb', pa.float32()),                                        # Deep-learning Real-Bogus score
    ('fid', pa.uint8()),                                          # Filter ID
    ('sciinpseeing', pa.float32()),                               # Seeing in science image
    ('magpsf', pa.float32()),                                     # Magnitude from PSF-fit
    ('sigmapsf', pa.float32()),                                   # 1-sigma uncertainty in magpsf
    ('classtar', pa.float32()),                                   # Star/Galaxy classification score
    ('ra', pa.float64()),                                         # Right Ascension
    ('dec', pa.float64()),                                        # Declination
    ('fwhm', pa.float32()),                                       # Seeing
    ('aimage', pa.float32()),                                     # Major axis
    ('bimage', pa.float32()),                                     # Minor axis
    ('elong', pa.float32()),                                      # Elongation
    ('nbad', pa.int32()),                                         # Number of bad pixels
    ('nneg', pa.int32()),                                         # Number of negative pixels
    ('jd', pa.float64()),                                         # Julian date
    ('ndethist', pa.uint32()),                                    # Number of detections in history
    ('ncovhist', pa.uint32()),                                    # Number of coverages in history
    ('jdstarthist', pa.float64()),                                # Julian date of first detection
])
PA_STRUCT_TYPE = pa.struct([pa.field(field.name, field.type) for field in PARQUET_SCHEMA])


def save_triplets_and_features_in_batches(records, output_folder, batch_size=1024):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    next_file_number = get_next_file_number(output_folder)
    output_file = output_folder / f"data_{next_file_number}.parquet"

    batch = []
    with pq.ParquetWriter(output_file, PARQUET_SCHEMA) as writer:
        for record in records:
            triplet_images = []
            correct_size = True
            for image_type in ['Science', 'Template', 'Difference']:
                fits_bytes = record[f'cutout{image_type}']['stampData']
                image = extract_fits_image(fits_bytes)
                if image.shape == IMAGE_SHAPE:
                    triplet_images.append(image)
                else:
                    correct_size = False
                    break

            if not correct_size:
                continue

            candidate = record.get('candidate', {})

            candidate_names = [name for name in PARQUET_SCHEMA.names if name != 'triplet' and name != 'objectId']
            batch_record = {
                'images': np.stack(triplet_images, axis=0).reshape(-1),
                'objectId': record.get('objectId', 'None'),
            } | {key: candidate.get(key, None) for key in candidate_names}

            batch.append(batch_record)

            if len(batch) == batch_size:
                save_batch(writer, batch)
                # Resetting the batch
                batch = []


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


def save_batch(parquet_writer, batch):
    struct_array = pa.array(batch, type=PA_STRUCT_TYPE)
    table = pa.Table.from_struct_array(struct_array)
    parquet_writer.write_table(table)


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
    parser.add_argument('--batch_size', type=int, default=1024, help="Number of AVRO files in each parquet row group")
    parser.add_argument('--min_batch_size', type=int, default=1024*100, help="Minimum number of AVRO files to trigger processing")

    args = parser.parse_args()

    main(args.start_date, args.end_date, args.extract_to, args.output_folder, args.min_file_size, args.batch_size, args.min_batch_size)



# EXAMPLE CODE
# python 0.ALL_DOWNLOAD.py --start_date 20240101 --end_date 20240421 --extract_to "D:/STAMP_AD_IMAGES/Data" --output_folder "D:/STAMP_AD_IMAGES/ProcessedData" --min_file_size 512 --batch_size 1024 --min_batch_size 102400
