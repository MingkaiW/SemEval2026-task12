import json
import os
import requests
import base64
import mimetypes
from urllib.parse import urlparse

# Configuration - Process all data splits
DATA_SPLITS = ['sample_data', 'train_data', 'dev_data', 'test_data']

def setup_directories(image_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

def determine_extension(header_or_url, is_base64=False, content_type=None):
    """Determines the appropriate file extension (.jpg, .png, etc.)"""
    if is_base64:
        # Example header: "data:image/jpeg;base64"
        mime = header_or_url.split(';')[0].split(':')[1]
        return mimetypes.guess_extension(mime) or '.jpg'
    else:
        # Try guessing from URL
        path = urlparse(header_or_url).path
        ext = os.path.splitext(path)[1]
        if ext:
            return ext
        # Fallback to Content-Type header from response
        if content_type:
            return mimetypes.guess_extension(content_type)
        return '.jpg'

def save_base64_image(data_string, save_path):
    try:
        # Split "data:image/jpeg;base64,......"
        header, encoded = data_string.split(',', 1)
        ext = determine_extension(header, is_base64=True)

        full_path = f"{save_path}{ext}"

        with open(full_path, "wb") as f:
            f.write(base64.b64decode(encoded))
        return full_path
    except Exception as e:
        print(f"Error saving Base64 image: {e}")
        return None

def download_url_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get('content-type')
            ext = determine_extension(url, is_base64=False, content_type=content_type)

            full_path = f"{save_path}{ext}"

            with open(full_path, "wb") as f:
                f.write(response.content)
            return full_path
        else:
            print(f"Failed to download {url} (Status: {response.status_code})")
            return None
    except Exception as e:
        print(f"Error downloading URL {url}: {e}")
        return None

def process_dataset(split_name):
    """Process a single data split"""
    input_file = os.path.join(split_name, 'docs.json')
    output_file = os.path.join(split_name, 'docs_updated.json')
    image_dir = os.path.join('downloaded_images', split_name)

    setup_directories(image_dir)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Could not find {input_file}. Skipping this split.")
        return False

    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()}: {len(dataset)} topics")
    print(f"{'='*60}")

    total_docs = 0
    saved_count = 0
    failed_count = 0

    for topic_idx, entry in enumerate(dataset):
        topic_id = entry.get('topic_id', 'unknown')
        topic_dir = os.path.join(image_dir, f"topic_{topic_id}")

        if not os.path.exists(topic_dir):
            os.makedirs(topic_dir)

        docs = entry.get('docs', [])
        total_docs += len(docs)
        print(f"  > Topic {topic_id}: Processing {len(docs)} documents...")

        for doc in docs:
            image_source = doc.get('imageUrl')
            doc_uuid = doc.get('id', 'no_id')

            # Define the base path for the image (extension added later)
            save_path_base = os.path.join(topic_dir, doc_uuid)

            saved_location = None

            if image_source:
                # CASE 1: Base64 Encoded Image
                if image_source.startswith('data:image'):
                    saved_location = save_base64_image(image_source, save_path_base)

                # CASE 2: Standard URL
                elif image_source.startswith('http'):
                    saved_location = download_url_image(image_source, save_path_base)

            # Upgrade the document info
            if saved_location:
                doc['local_image_path'] = saved_location
                saved_count += 1
                print(f"    + Saved: {doc_uuid}")
            else:
                doc['local_image_path'] = None
                if image_source:
                    failed_count += 1
                    print(f"    ! Failed to save image for {doc_uuid}")

    # Save the upgraded dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

    print(f"\n{split_name.upper()} Summary:")
    print(f"  Total documents: {total_docs}")
    print(f"  Images saved: {saved_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Images directory: {image_dir}/")
    print(f"  Updated dataset: {output_file}")

    return True

def process_all_splits():
    """Process all data splits"""
    print("Starting image processing for all data splits...")

    processed_splits = []
    for split in DATA_SPLITS:
        if process_dataset(split):
            processed_splits.append(split)

    print(f"\n{'='*60}")
    print("ALL PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed splits: {', '.join(processed_splits)}")
    print("All images saved to: downloaded_images/")
    print("Updated datasets saved in respective split directories")

if __name__ == "__main__":
    process_all_splits()
