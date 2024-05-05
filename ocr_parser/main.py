

# Import the required modules
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from pathlib import Path
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
import numpy as np
import os
import cv2

def normalize_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)


    if image is None:

        raise ValueError(f"Unable to load image from path: {image_path}")

    # Normalize colors
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

    # Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Skew correction
    coords = np.column_stack(np.where(image > 0))
    angle = 90 - cv2.minAreaRect(coords)[-1]
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Specify the Google Drive folder path
    drive_path = "./images/"
    # Create the folder if it doesn't exist
    # Path(drive_path).mkdir(parents=True, exist_ok=True)

    normalized_image_path = f"{drive_path}" + os.path.basename(image_path)
    cv2.imwrite(normalized_image_path, image)

    return normalized_image_path

def doctr_ocr(path, name):
    model = ocr_predictor(pretrained=True)

    # Normalize the image
    # normalized_image_path = normalize_image(path)

    # doc = DocumentFile.from_images(normalized_image_path)
    doc = DocumentFile.from_images(path)
    result = model(doc)
    # result.show(doc)
    json_response = result.export()
    file_path = "./outputs/"+ name + ".txt"
    with open(file_path, "w+") as file:
        pass
    for block in json_response["pages"][0]["blocks"]:
        for line in block["lines"]:
            val = ""
            for word in line["words"]:
                val += word["value"]
                val += " "
            val += "\n"
            with open(file_path, "a") as file:
                file.write(val)

folder_path = "images"
file_list = os.listdir(folder_path)

for path in file_list:
    print(path)


  # Construct the full path to the image
    # full_path = os.path.join(folder_path, path)
    # drive_folder ="processedlicense"

    if path != ".ipynb_checkpoints":
        doctr_ocr(folder_path + "/" + path, path)
    else:
        pass

def aadhar(text):
    # Define regular expressions for different pieces of information
    name_pattern = re.compile(r'Name[\n\s]+(.+?)[\n\s]+(?:DOB|Year of Birth|DO8)', re.DOTALL | re.IGNORECASE)
    dob_pattern = re.compile(r'(DOB[:/ ](\d{2}/\d{2}/\d{4}|\d{8}|\d{4})|Year of Birth*[:/ ]*(\d{4})|DO8[:/ ]*(\d{2}/\d{2}/\d{4}|\d{8}|\d{4}))', re.DOTALL | re.IGNORECASE)
    gender_pattern = re.compile(r'(MALE|FEMALE|TRANSGENDER|OTHERS?)[\n\s]+(.+?)[\n\s]+(\d{4} \d{4} \d{4})?', re.DOTALL | re.IGNORECASE)
    aadhar_number_pattern = re.compile(r'(\d{4} \d{4} \d{4})', re.DOTALL)

    # Extract information using regular expressions
    name_match = name_pattern.search(text)
    dob_match = dob_pattern.search(text)
    gender_match = gender_pattern.search(text)
    aadhar_number_match = aadhar_number_pattern.search(text)

    lines = text.split('\n')

    prev_line = ""
    name = ""

    for i in range(1, len(lines)):

      dob_match1 = dob_pattern.search(lines[i])
      if dob_match1:
        name = lines[i-1]

    # Extracted information
    extracted_info = {
        'Name': name,
        'Date of Birth': dob_match.group(2) or dob_match.group(3) or dob_match.group(4) if dob_match else None,
        'Gender': gender_match.group(1).strip() if gender_match else None,
        'Aadhar Number': aadhar_number_match.group(1).replace(" ", "") if aadhar_number_match else None,
    }

    return extracted_info

def pan(text):
    # Define regular expressions for different pieces of information
    name_pattern = re.compile(r'Name[\n\s]+(.+?)(?:\n|$)', re.DOTALL)
    father_name_pattern = re.compile(r'Father\'s Name[\n\s]+(.+?)(?:\n|$)', re.DOTALL)
    pan_number_pattern = re.compile(r'Permanent Account Number Card[\n\s]+(.+?)(?:\n|$)')
    dob_pattern = re.compile(r'Date of Birth[\n\s]+(.+?)(?:\n|$)')

    # Extract information using regular expressions
    name_match = name_pattern.search(text)
    father_name_match = father_name_pattern.search(text)
    pan_number_match = pan_number_pattern.search(text)
    dob_match = dob_pattern.search(text)

    # Extracted information
    extracted_info = {
        'Name': ' '.join(name_match.group(1).strip().split()) if name_match else None,
        'Father\'s Name': ' '.join(father_name_match.group(1).strip().split()) if father_name_match else None,
        'PAN Number': pan_number_match.group(1) if pan_number_match else None,
        'DOB': dob_match.group(1) if dob_match else None,
    }

    return extracted_info

def cbse(text):
    # Define regular expressions for different pieces of information
    roll_number_pattern = re.compile(r'Roll No[.:-]? (\d+)')
    unique_id_pattern = re.compile(r'UNIQUE ID[.:-]? (\d+)')
    name_pattern = re.compile(r'Name(?: of Candidate)?[.-: ]? (\w+ \w+)')
    school_pattern = re.compile(r'\nSchool[.:-]?[\']?[ ]*[a-zA-z ]*[:]?\d*([A-Za-z,:\' ]+)')
    school_pattern2 = re.compile(r'of ([A-Za-z,\' ]+)')
    # marks_pattern = re.compile(r'(\d{3}) ([A-Z &]+)\n(\d{3}) (\d{2,3})')
    marks_pattern = re.compile(r'(\d{3}) ([A-Z &]+)\n(?:\d{3} )?(\d{2,3})?')

    issue_date = re.compile(r'date[a-zA-Z: ]* ((0[1-9]|[12][0-9]|3[01])[-.](0[1-9]|1[0,1,2])[-.](19|20)\d{2})', re.IGNORECASE)

    # Extract information using regular expressions
    roll_number_match = roll_number_pattern.search(text)
    unique_id_match = unique_id_pattern.search(text)
    name_match = name_pattern.search(text)
    school_match = school_pattern.search(text)
    school_match2 = school_pattern2.search(text)
    marks_matches = marks_pattern.findall(text)
    issue_date_match = issue_date.search(text)

    # Create a dictionary to store extracted information
    extracted_info = {
        'Roll Number / Unique ID': roll_number_match.group(1) if roll_number_match else unique_id_match.group(1) if unique_id_match else None,
        'Name': name_match.group(1) if name_match else None,
        'School': school_match.group(1).strip() if school_match else school_match2.group(1).strip() if school_match2 else None,
        'Marks': {subject: (theory, practical) for theory, subject, practical in marks_matches} if marks_matches else None,
        'Issue date': issue_date_match.group(1) if issue_date_match else None
    }

    return extracted_info

import os
import re

def parse_aadhaar(file_path):
    # Custom parser for Aadhaar files
    with open(file_path, 'r') as file:
        text = file.read()
        result = aadhar(text)
    print("Information from aadhar:")
    print(result)


def parse_pan(file_path):
    # Your custom parser for PAN files
    with open(file_path, 'r') as file:
        text = file.read()
        result = pan(text)
    print("Information from PAN:")
    print(result)



def parse_cbse(file_path):
    # Your custom parser for PAN files
    with open(file_path, 'r') as file:
        text = file.read()
        result = cbse(text)
    print("Information from CBSE:")
    print(result)



def parse_icse(file_path):
    # Your custom parser for PAN files
    with open(file_path, 'r') as file:
        text = file.read()
        result = cbse(text)
    print("Information from ICSE:")
    print(result)



def parse_other(file_path):
    # Your custom parser for other files
    print(f"Parsing other file: {file_path}")



def choose_parser(file_path):
    filename = os.path.basename(file_path)
    keywords = ['aadhar', 'aadhaar', 'pan', 'cbse', 'icse']  # Add more keywords as needed

    for keyword in keywords:
        regex_pattern = re.compile(rf'\b{keyword}\b', re.IGNORECASE)
        if regex_pattern.search(filename):
            if 'aadhar' in keyword.lower():
                parse_aadhaar(file_path)
            elif 'pan' in keyword.lower():
                parse_pan(file_path)
            elif 'cbse' in keyword:
                parse_cbse(file_path)
            elif 'icse' in keyword:
                parse_icse(file_path)
            else:
                parse_other(file_path)
            break
    else:
        parse_other(file_path)

# Directory where your files are stored
directory_path = './outputs/'

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        choose_parser(file_path)