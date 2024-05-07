from datetime import datetime, timedelta
import tarfile
import argparse
import random
import fastavro
import numpy as np
import os
import stat
from pathlib import Path
import gzip
from astropy.io import fits
import io
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import tensorflow as tf


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


import time

def safe_extract_tarfile(filepath, extract_to, max_retries=3, delay=2):
    """
    Safely extracts a tar.gz file to the specified directory, handling exceptions and applying retry logic.
    This function will retry extraction up to a specified number of retries and includes a delay between retries
    for cases like tar file errors.

    :param filepath: Path to the tar.gz file to be extracted.
    :param extract_to: Destination directory where files will be extracted.
    :param max_retries: Maximum number of retry attempts for extraction.
    :param delay: Delay in seconds between retry attempts.
    """
    with tarfile.open(filepath, "r:gz") as tar:
        successful_extractions = 0
        failed_extractions = 0
        for member in tar.getmembers():
            retries = 0
            while retries <= max_retries:
                try:
                    tar.extract(member, path=extract_to)
                    successful_extractions += 1
                    break  # Success: exit the retry loop
                except PermissionError as e:
                    print(f"Permission error occurred while extracting {member.name}: {e}. Skipping this file.")
                    failed_extractions += 1
                    break  # Stop retries for permission errors as these are unlikely to be transient
                except tarfile.TarError as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"Tar file error with {member.name}: {e}. Skipping this file after {max_retries} retries.")
                        failed_extractions += 1
                    else:
                        print(f"Retry {retries}/{max_retries} for {member.name} due to tar error: {e}.")
                        time.sleep(delay)  # Wait before retrying
                except Exception as e:
                    print(f"An unexpected error occurred while extracting {member.name}: {e}. Skipping this file.")
                    failed_extractions += 1
                    break  # Stop retries for unknown exceptions

        print(f"Completed extracting {successful_extractions} files successfully.")
        if failed_extractions > 0:
            print(f"Failed to extract {failed_extractions} files due to errors.")


def download_and_unzip(url, extract_to, min_file_size, success_log, error_log):
    """
    Attempts to download and extract a tar.gz file from a URL. It implements retry logic for robustness
    against network-related errors and ensures clean handling of file operations.

    Parameters:
    - url: The URL from which to download the file.
    - extract_to: Directory path where the extracted contents should be stored.
    - min_file_size: Minimum file size required to proceed with extraction.
    - success_log: Path to the file where successful operations are logged.
    - error_log: Path to the file where errors are logged.
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)

    # Setup retry strategy for robust network operation
    retry_strategy = Retry(
        total=5,  # Total number of retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # HTTP methods to retry
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        with session.get(url, stream=True, timeout=10) as response:  # Set timeout to 10 seconds
            response.raise_for_status()
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        # Check if the downloaded file meets the size requirement before processing
        if os.path.exists(filepath) and os.path.getsize(filepath) > min_file_size:
            safe_extract_tarfile(filepath, extract_to)
            os.remove(filepath)
            with open(success_log, 'a') as log:
                log.write(f"Successfully downloaded and extracted: {url}\n")
            return True
        else:
            if os.path.exists(filepath):
                os.remove(filepath)
            with open(error_log, 'a') as log:
                log.write(f"File {filename} was too small or not a tar.gz and has been removed.\n")
            return False
    except requests.exceptions.RequestException as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(error_log, 'a') as log:
            log.write(f"Failed to download {url}: {e}\n")
        return False

def shuffle_avro_file_paths(folder_path, seed=42):
    """
    Shuffles and returns a list of .avro file paths located in the specified directory.
    This function is useful for randomizing the order of files before processing to mitigate
    any potential bias or ordering effects inherent in the file system.

    :param folder_path: Directory containing .avro files.
    :param seed: Random seed for reproducibility of the shuffle order.
    :return: List of shuffled .avro file paths.
    """
    random.seed(seed)
    file_paths = list(Path(folder_path).glob('*.avro'))
    random.shuffle(file_paths)
    return file_paths

def read_avro_files(file_paths):
    """
    Generator function that reads .avro files from a list of file paths and yields Avro records.
    This function facilitates the reading of multiple Avro files, allowing for the processing
    of large datasets spread across several files.

    :param file_paths: Iterable of .avro file paths to be read.
    :yield: Yields one record at a time from the .avro files.
    """
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            reader = fastavro.reader(f)
            for record in reader:
                # Optionally add the file path to the record if necessary
                record['file_path'] = str(file_path)
                yield record


def extract_and_process_image(fits_bytes, issdiffpos):
    """
    Processes FITS image data by reading from a compressed byte array and optionally inverting the image data.
    This function is specifically tailored for astronomical data where image inversion may be required based
    on specific conditions (e.g., 'issdiffpos' flag).

    :param fits_bytes: Compressed byte array containing FITS image data.
    :param issdiffpos: Flag indicating whether the image data needs to be inverted.
    :return: Processed image data as a numpy array.
    """
    with gzip.open(io.BytesIO(fits_bytes), 'rb') as gz:
        with fits.open(io.BytesIO(gz.read())) as hdul:
            image_data = hdul[0].data.astype(np.float32)
            # Check if issdiffpos is 'f', and if so, invert the image data
            if issdiffpos == 'f':
                image_data *= -1
            return image_data


def get_next_file_number(output_folder):
    """
    Determines the next file number to be used for saving new TFRecord files in a specified directory.
    This function avoids overwriting existing files by incrementing the file number based on the existing files.

    :param output_folder: Directory where the TFRecord files are saved.
    :return: Next file number to be used for a new TFRecord file.
    """
    existing_files = list(Path(output_folder).glob('data_*.tfrecord'))
    if not existing_files:
        return 0
    else:
        max_file_num = max(int(file.stem.split('_')[-1]) for file in existing_files)
        return max_file_num + 1


def save_triplets_and_features_in_batches(records, output_folder, batch_size, success_log, error_log):
    """
    Saves batches of triplets and features into TFRecord files.
    Returns paths of successfully processed files and any remaining records if they don't fill a full batch.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    batch_images = []
    batch_objectIds = []
    batch_candids = []
    batch_other_features = []
    successfully_saved_files = []
    remaining_records = []

    file_number = get_next_file_number(output_folder)

    for record in records:
        triplet_images = []
        correct_size = True
        for image_type in ['Science', 'Template', 'Difference']:
            fits_bytes = record[f'cutout{image_type}']['stampData']
            issdiffpos = record.get('issdiffpos', 't')  # Assume 't' as default
            image = extract_and_process_image(fits_bytes, issdiffpos if image_type == 'Difference' else 't')
            if image.shape == (63, 63):
                triplet_images.append(image)
            else:
                correct_size = False
                break

        if correct_size:
            candidate = record.get('candidate', {})
            objectId = record.get('objectId', 'NoObjectId')
            candid = int(candidate.get('candid', 0))
            numeric_features = [float(candidate.get(k, np.nan)) for k in ['rb', 'drb', 'sciinpseeing', 'magpsf', 'sigmapsf', 'classtar', 'ra', 'dec', 'fwhm', 'aimage', 'bimage', 'elong', 'nbad', 'nneg', 'jd', 'ndethist', 'ncovhist', 'jdstarthist', 'jdendhist']]
            numeric_features.append(1 if candidate.get('isdiffpos', 't') == 't' else 0)
            batch_images.append(np.stack(triplet_images, axis=0))
            batch_objectIds.append(objectId)
            batch_candids.append(candid)
            batch_other_features.append(numeric_features)

            if len(batch_images) == batch_size:
                if save_batch_and_log(output_folder, file_number, batch_images, batch_objectIds, batch_candids, batch_other_features, success_log, error_log):
                    successfully_saved_files.extend([record['file_path'] for record in records])
                batch_images, batch_objectIds, batch_candids, batch_other_features = [], [], [], []
                file_number += 1
        else:
            remaining_records.append(record)

    # Handle any remaining records that didn't complete a full batch
    if len(batch_images) >= batch_size:
        if save_batch_and_log(output_folder, file_number, batch_images, batch_objectIds, batch_candids, batch_other_features, success_log, error_log):
            successfully_saved_files.extend([record['file_path'] for record in records])
        batch_images, batch_objectIds, batch_candids, batch_other_features = [], [], [], []
        file_number += 1

    return successfully_saved_files, remaining_records


def safe_remove(file_path):
    """
    Safely removes a file from the filesystem, handling permissions and other exceptions.
    This function attempts to change the file's permissions to writable before attempting deletion,
    catching and reporting any errors encountered during the process. This is particularly useful
    in environments where file permissions might prevent deletion, ensuring robust file management.

    :param file_path: Path to the file to be removed.
    """
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


def process_and_cleanup_avro_batch(folder_path, output_folder, batch_size, min_avro_files, success_log, error_log):
    """
    Processes .avro files into TFRecords if they meet the minimum batch size requirement.
    """
    avro_file_paths = shuffle_avro_file_paths(folder_path)
    if len(avro_file_paths) < min_avro_files:
        print("Not enough .avro files to start processing.")
        return

    records = read_avro_files(avro_file_paths)
    print('Processing Records...')
    successfully_saved_files, remaining_records = save_triplets_and_features_in_batches(
        records, output_folder, batch_size, success_log, error_log)
    print('Done Saving Triplets. Cleaning up processed AVRO files...')

    # Cleanup successfully processed files
    for file_path in successfully_saved_files:
        safe_remove(file_path)
    print('Cleanup completed.')

    # If remaining records still meet the minimum batch size, process them
    if len(remaining_records) >= min_avro_files:
        additional_saved_files, _ = save_triplets_and_features_in_batches(
            remaining_records, output_folder, batch_size, success_log, error_log)
        for file_path in additional_saved_files:
            safe_remove(file_path)


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    elif isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(images, objectIds, candids, features):
    feature = {
        'images': _bytes_feature(tf.io.serialize_tensor(images).numpy()),
        'objectIds': _bytes_feature(objectIds),  # Object ID as a string, converted to bytes
        'candids': _int64_feature(candids),
        'features': _bytes_feature(tf.io.serialize_tensor(features).numpy()),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def save_batch_and_log(output_folder, file_number, batch_images, batch_objectIds, batch_candids, batch_other_features, success_log, error_log):
    """
    Serializes batches of images and features into TFRecord format and saves them to disk,
    incrementing the file number for each new batch to prevent overwriting.
    """
    filename = f"{output_folder}/data_{file_number}.tfrecord"
    try:
        with tf.io.TFRecordWriter(filename) as writer:
            for images, objectId, candid, features in zip(batch_images, batch_objectIds, batch_candids, batch_other_features):
                example = serialize_example(images, objectId, candid, features)
                writer.write(example)
        with open(success_log, 'a') as log:
            log.write(f"Saved batch in {filename}\n")
        return True
    except Exception as e:
        with open(error_log, 'a') as log:
            log.write(f"Failed to save batch {filename}: {e}\n")
        return False


def main(start_date, end_date, extract_to, output_folder, min_file_size, batch_size, min_avro_files):
    url_template = 'https://ztf.uw.edu/alerts/public/ztf_public_{date}.tar.gz'
    urls = generate_date_urls(start_date, end_date, url_template)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize log files in the specified output directory
    success_log = os.path.join(output_folder, "success_log.txt")
    error_log = os.path.join(output_folder, "error_log.txt")

    # Open and close the log files to ensure they exist
    open(success_log, 'a').close()
    open(error_log, 'a').close()

    for url in urls:
        # Pass the log file paths as arguments to the download_and_unzip function
        success = download_and_unzip(url, extract_to, min_file_size, success_log, error_log)
        if not success:
            continue  # Skip to the next URL if extraction failed

        avro_files = list(Path(extract_to).glob('*.avro'))
        if len(avro_files) >= min_avro_files:
            process_and_cleanup_avro_batch(extract_to, output_folder, batch_size, min_avro_files, success_log, error_log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download, process, and manage AVRO files efficiently.")
    parser.add_argument('--start_date', required=True, help="Start date in 'YYYYMMDD' format")
    parser.add_argument('--end_date', required=True, help="End date in 'YYYYMMDD' format")
    parser.add_argument('--extract_to', required=True, help="Directory where tar.gz contents will be extracted")
    parser.add_argument('--output_folder', required=True, help="Directory where processed data will be saved")
    parser.add_argument('--min_file_size', type=int, default=512, help="Minimum file size in bytes to proceed with extraction")
    parser.add_argument('--batch_size', type=int, default=2048, help="Number of AVRO files in each TFRecord file")
    parser.add_argument('--min_avro_files', type=int, default=204800, help="Minimum number of AVRO files to trigger processing")

    args = parser.parse_args()
    main(args.start_date, args.end_date, args.extract_to, args.output_folder, args.min_file_size, args.batch_size, args.min_avro_files)



# EXAMPLE CODE
# python 0.ALL_DOWNLOAD_TFRecords.py --start_date 20240101 --end_date 20240501 --extract_to "/home/fiore/Desktop/Data" --output_folder "/home/fiore/Desktop/ProcessedData_TFRecords" --min_file_size 512 --batch_size 2048 --min_avro_files 204800




# import tensorflow as tf
#
#
# def parse_tfrecord(example_proto):
#     # Define the expected structure of your TFRecord data
#     feature_description = {
#         'images': tf.io.FixedLenFeature([], tf.string),  # Images stored as raw byte strings
#         'objectIds': tf.io.FixedLenFeature([], tf.string),  # Object IDs stored as byte strings
#         'candids': tf.io.FixedLenFeature([], tf.int64),  # Candid numbers as int64
#         'features': tf.io.FixedLenFeature([], tf.string),  # Features stored as raw byte strings
#     }
#     # Parse the input tf.train.Example proto using the dictionary above
#     features = tf.io.parse_single_example(example_proto, feature_description)
#
#     # Decode the images and features correctly
#     images = tf.io.parse_tensor(features['images'], out_type=tf.float32)  # Assuming images were saved as float32
#     images = tf.reshape(images, shape=(-1, 63, 63))  # Adjust shape as necessary
#
#     # Here, ensure the type matches how features were stored. If they were stored as float64, parse as float64:
#     features_decoded = tf.io.parse_tensor(features['features'], out_type=tf.float64)  # Adjust if stored as float64
#
#     # Return images and other metadata
#     return images, features['objectIds'], features['candids'], features_decoded
#
#
# def read_tfrecords(file_path):
#     # Create a dataset object from the TFRecord file
#     dataset = tf.data.TFRecordDataset(file_path)
#
#     # Map the parsing function to each element in the dataset
#     dataset = dataset.map(parse_tfrecord)
#
#     return dataset
#
#
# # Specify the path to your TFRecord file
# tfrecord_file = '/ProcessedData_TFRecords/data_0.tfrecord'
#
# # Read the TFRecord file
# dataset = read_tfrecords(tfrecord_file)
#
#
# import matplotlib.pyplot as plt
#
# # Iterate over the first few records and print the content
# for images, objectIds, candids, features in dataset.take(1):  # Adjust `.take()` for more samples
#     # print("Image shape:", images.shape)
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)  # Use figsize to adjust overall size
#     for i in range(3):
#         axs[i].imshow(images[i, :, :])  # Set colormap to 'gray' if it suits the image data better
#
#     plt.tight_layout()  # Adjust layout to prevent overlap
#     plt.show()  # Display the plots
#     print("Object ID:", objectIds.numpy().decode('utf-8'))
#     print("Candid:", candids.numpy())
#     print("Features shape:", features.shape)
#     print("Sample features:", features.numpy())  # Print first few elements of features
#


