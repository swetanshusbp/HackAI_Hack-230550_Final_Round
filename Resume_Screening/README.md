# README 

# For Resume Screening Project
### Project Description
The Resume Screening project is a Python-based application that allows users to classify resumes into different categories. It uses machine learning to analyze resumes, categorizing them based on content. The application is designed to handle PDF files, which are processed to extract text and then classified.

### Prerequisites

Python (3.6 or higher recommended)
Required Python Libraries: numpy, pandas, nltk, scikit-learn, pdfminer.six
A dataset of resumes in a CSV format

### Installation Instructions

### Clone the Repository:

Clone the project repository to your local machine.

### Install Python Libraries:
### Install the required Python libraries using pip:

pip install numpy pandas nltk scikit-learn pdfminer.six

### Set up NLTK Resources:
### Download the necessary NLTK resources:

import nltk
nltk.download('stopwords')
Running the Project
Prepare Your Dataset:
Ensure your dataset of resumes is in CSV format and is accessible to the script.

### Run the Script:
### Execute the main script:

python main.py
Replace main.py with the name of your Python script.

### Using the Application:

The script allows you to input the path of a PDF resume.
Ensure that the PDF resume is saved on your system and its path is correctly specified.
The script will process the resume and output its predicted category.

### Special Considerations
Ensure that the PDF files you wish to analyze are accessible to the script(It has to be saved in the system and the system path ahs to be used).
The accuracy of predictions depends on the quality of the dataset used for training the model.

![WhatsApp Image 2023-12-27 at 18 50 09_dbd66994](https://github.com/swetanshusbp/HackAI_Hack-230550_Final_Round/assets/97033991/69946718-6e12-4cb9-9713-30b7e0587b0b)

