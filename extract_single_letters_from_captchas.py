import os
import os.path
import cv2
import glob
import imutils

CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# Loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Extract the base filename as the captcha text (e.g., "2A2X.png" -> "2A2X")
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # Apply a binary threshold to the image (convert it to black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find the contours (continuous blobs of pixels) in the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Loop through each contour to extract the letters
    for contour in contours:
        # Get the bounding rectangle for the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Detect letters that are too wide (likely conjoined letters)
        if w / h > 1.25:
            # Split the wide contour into two letter regions
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a single letter region
            letter_image_regions.append((x, y, w, h))

    # Ensure we have exactly 4 letters; otherwise, skip this image
    if len(letter_image_regions) != 4:
        continue

    # Sort the detected letter images based on the x coordinate (left to right)
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save each letter as a separate image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Get the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Define the path to save the letter image
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # Create the output directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # Increment the count for the current letter
        counts[letter_text] = count + 1
