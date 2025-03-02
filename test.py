#!python

import os
import argparse
import tensorflow as tf

def resize_images_in_folder(folder_path,  suffix="_sr.png"):
    """
    Resizes all images in a folder and saves them with a suffix.

    Args:
        folder_path: Path to the folder containing images.
        output_size: Tuple (height, width) for the resized images.
        suffix: Suffix to add to the resized image filenames.
    """

    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

    output_dir = os.path.join(folder_path, 'sr')
    os.makedirs(output_dir, exist_ok=True)  #

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')  # Add more if needed
    processed_count = 0

    model = tf.saved_model.load("saved_model")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            try:
                image_path = os.path.join(folder_path, filename)
                input_image = tf.keras.utils.img_to_array(tf.keras.utils.load_img(image_path))
                input_image = input_image / 255.0
                input_image = tf.expand_dims(input_image, axis=0)
                output_image = model.serve(input_image)
                output_image = tf.image.convert_image_dtype(output_image, dtype=tf.uint8, saturate=True)
                output_image = tf.squeeze(output_image, axis=0)
                print(output_image.shape)
                base, ext = os.path.splitext(filename)
                output_filename = f"{base}{suffix}"
                output_path = os.path.join(output_dir, output_filename)
                tf.keras.utils.save_img(output_path, output_image, scale=False)
                print(f"Resized and saved: {filename} -> {output_filename}")
                processed_count += 1

            except (tf.errors.InvalidArgumentError, OSError) as e:
                print(f"Error processing {filename}: {e}")
            except Exception as e: #catch all other exception
                print(f"Unexpected error processing {filename}: {e}")

    print(f"Processed {processed_count} images.")


def main():
    parser = argparse.ArgumentParser(description="Resize images in a folder.")
    parser.add_argument("input_dir", type=str, help="Path to the folder containing images.")
    parser.add_argument("--suffix", type=str, default="_sr.png",
                        help="Suffix for resized images. Default: _sr.png")

    args = parser.parse_args()

    try:
        resize_images_in_folder(args.input_dir, args.suffix)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()