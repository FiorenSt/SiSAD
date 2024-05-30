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
    with tarfile.open(filepath, "r:gz") as tar:
        for member in tar.getmembers():
            try:
                tar.extract(member, path=extract_to)
            except PermissionError as e:
                print(f"Permission error occurred while extracting {member.name}: {e}. Skipping this file.")
            except Exception as e:
                print(f"An unexpected error occurred while extracting {member.name}: {e}. Skipping this file.")


def download_and_unzip(url, extract_to, min_file_size, success_log, error_log):
    """
    Download and extract a single tar.gz file from the given URL with retry logic for robustness.
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)

    # Set up retry strategy
    retry_strategy = Retry(
        total=5,  # total number of retries
        backoff_factor=1,  # exponential backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # retry on these status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # only retry on these HTTP methods
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        with session.get(url, stream=True) as response:
            response.raise_for_status()
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        if os.path.getsize(filepath) > min_file_size and filename.endswith("tar.gz"):
            safe_extract_tarfile(filepath, extract_to)
            os.remove(filepath)
            with open(success_log, 'a') as log:
                log.write(f"Successfully downloaded and extracted: {url}\n")
            return True
        else:
            os.remove(filepath)
            with open(error_log, 'a') as log:
                log.write(f"File {filename} was too small or not a tar.gz and has been removed.\n")
            return False
    except Exception as e:
        os.remove(filepath)
        with open(error_log, 'a') as log:
            log.write(f"An unexpected error occurred with {filename}: {e}\n")
        return False

def shuffle_avro_file_paths(folder_path, seed=42):
    random.seed(seed)
    file_paths = list(Path(folder_path).glob('*.avro'))
    random.shuffle(file_paths)
    return file_paths

def read_avro_files(file_paths):
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            reader = fastavro.reader(f)
            for record in reader:
                yield record

def extract_and_process_image(fits_bytes, issdiffpos):
    with gzip.open(io.BytesIO(fits_bytes), 'rb') as gz:
        with fits.open(io.BytesIO(gz.read())) as hdul:
            image_data = hdul[0].data.astype(np.float32)
            # Check if issdiffpos is 'f', and if so, invert the image data
            if issdiffpos == 'f':
                image_data *= -1
            return image_data


def get_next_file_number(output_folder):
    existing_files = list(Path(output_folder).glob('data_*.h5'))
    return 0 if not existing_files else max(int(file.stem.split('_')[-1]) for file in existing_files) + 1

def save_triplets_and_features_in_batches(records, output_folder, batch_size, success_log, error_log):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    batch_images = []
    batch_objectIds = []
    batch_candids = []
    batch_other_features = []
    file_number = get_next_file_number(output_folder)

    for record in records:
        triplet_images = []
        correct_size = True
        for image_type in ['Science', 'Template', 'Difference']:
            fits_bytes = record[f'cutout{image_type}']['stampData']
            # Extract issdiffpos and convert it directly in the call
            issdiffpos = record.get('issdiffpos',  np.nan)  # Assume 't' as default if missing
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
            numeric_features = [
                float(candidate.get('rb', np.nan)),
                float(candidate.get('drb', np.nan)),
                int(candidate.get('fid', np.nan)),
                float(candidate.get('sciinpseeing', np.nan)),
                float(candidate.get('magpsf', np.nan)),
                float(candidate.get('sigmapsf', np.nan)),
                float(candidate.get('classtar', np.nan)),
                float(candidate.get('ra', np.nan)),
                float(candidate.get('dec', np.nan)),
                float(candidate.get('fwhm', np.nan)),
                float(candidate.get('aimage', np.nan)),
                float(candidate.get('bimage', np.nan)),
                float(candidate.get('elong', np.nan)),
                int(candidate.get('nbad', np.nan)),
                int(candidate.get('nneg', np.nan)),
                float(candidate.get('jd', np.nan)),
                int(candidate.get('ndethist', np.nan)),
                int(candidate.get('ncovhist', np.nan)),
                float(candidate.get('jdstarthist', np.nan)),
                float(candidate.get('jdendhist', np.nan)),
                1 if candidate.get('isdiffpos',  np.nan) == 't' else 0  # Convert 't'/'f' to 1/0
            ]
            batch_images.append(np.stack(triplet_images, axis=0))
            batch_objectIds.append(objectId)
            batch_candids.append(candid)
            batch_other_features.append(numeric_features)

            if len(batch_images) == batch_size:
                save_batch_and_log(output_folder, file_number, batch_images, batch_objectIds, batch_candids, batch_other_features, success_log, error_log)
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



def process_and_cleanup_avro_batch(folder_path, output_folder, batch_size, min_batch_size, success_log, error_log):
    """
    Processes a batch of AVRO files into H5 and then cleans up the AVRO files.
    """
    avro_file_paths = shuffle_avro_file_paths(folder_path)
    if len(avro_file_paths) >= min_batch_size:
        records = read_avro_files(avro_file_paths[:min_batch_size])
        print('Processing Records...')
        save_triplets_and_features_in_batches(records, output_folder, batch_size, success_log, error_log)
        print('Done Saving Triplets. Cleaning up processed AVRO files...')
        for file_path in avro_file_paths[:min_batch_size]:
            safe_remove(file_path)  # Use safe_remove instead of os.remove
        print('Cleanup completed.')


import tensorflow as tf

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
    try:
        filename = f"{output_folder}/data_{file_number}.tfrecord"
        with tf.io.TFRecordWriter(filename) as writer:
            for images, objectId, candid, features in zip(batch_images, batch_objectIds, batch_candids, batch_other_features):
                example = serialize_example(np.array(images), objectId, candid, np.array(features))
                writer.write(example)
        with open(success_log, 'a') as log:
            log.write(f"Saved batch in data_{file_number}.tfrecord\n")
    except Exception as e:
        with open(error_log, 'a') as log:
            log.write(f"Failed to save batch data_{file_number}.tfrecord: {e}\n")



def main(start_date, end_date, extract_to, output_folder, min_file_size, batch_size, min_batch_size):
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
        if len(avro_files) >= min_batch_size:
            process_and_cleanup_avro_batch(extract_to, output_folder, batch_size, min_batch_size, success_log, error_log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download, process, and manage AVRO files efficiently.")
    parser.add_argument('--start_date', required=True, help="Start date in 'YYYYMMDD' format")
    parser.add_argument('--end_date', required=True, help="End date in 'YYYYMMDD' format")
    parser.add_argument('--extract_to', required=True, help="Directory where tar.gz contents will be extracted")
    parser.add_argument('--output_folder', required=True, help="Directory where processed data will be saved")
    parser.add_argument('--min_file_size', type=int, default=512, help="Minimum file size in bytes to proceed with extraction")
    parser.add_argument('--batch_size', type=int, default=20480, help="Number of AVRO files in each h5 file")
    parser.add_argument('--min_batch_size', type=int, default=204800, help="Minimum number of AVRO files to trigger processing")

    args = parser.parse_args()
    main(args.start_date, args.end_date, args.extract_to, args.output_folder, args.min_file_size, args.batch_size, args.min_batch_size)



# EXAMPLE CODE
# python 0.ALL_DOWNLOAD_TFRecords.py --start_date 20240101 --end_date 20240421 --extract_to "D:/STAMP_AD_IMAGES/Data" --output_folder "D:/STAMP_AD_IMAGES/ProcessedData_TFRecords" --min_file_size 512 --batch_size 2048 --min_batch_size 10240




# import tensorflow as tf


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

#     # Decode the images and features correctly
#     images = tf.io.parse_tensor(features['images'], out_type=tf.float32)  # Assuming images were saved as float32
#     images = tf.reshape(images, shape=(3, 63, 63))  # Adjust shape as necessary

#     # Here, ensure the type matches how features were stored. If they were stored as float64, parse as float64:
#     features_decoded = tf.io.parse_tensor(features['features'], out_type=tf.float64)  # Adjust if stored as float64

#     # Return images and other metadata
#     return images, features['objectIds'], features['candids'], features_decoded


# def read_tfrecords(file_path):
#     # Create a dataset object from the TFRecord file
#     dataset = tf.data.TFRecordDataset(file_path)

#     # Map the parsing function to each element in the dataset
#     dataset = dataset.map(parse_tfrecord)

#     return dataset


# # Specify the path to your TFRecord file
# tfrecord_file = 'D:/STAMP_AD_IMAGES/ProcessedData_TFRecords_GOOD/data_0.tfrecord'

# # Read the TFRecord file
# dataset = read_tfrecords(tfrecord_file)


# import matplotlib.pyplot as plt

# # Iterate over the first few records and print the content
# for images, objectIds, candids, features in dataset.take(30):  # Adjust `.take()` for more samples
#     print("Image shape:", images.shape)
#     plt.figure()
#     plt.imshow(images[2,:,:])
#     # print("Object ID:", objectIds.numpy().decode('utf-8'))
#     # print("Candid:", candids.numpy())
#     # print("Features shape:", features.shape)
#     # print("Sample features:", features.numpy()[:5])  # Print first few elements of features
