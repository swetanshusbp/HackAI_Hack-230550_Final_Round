# README 

# For Resume Screening Project
### Project Description
The Resume Screening project is a Python-based application that allows users to classify resumes into different categories. It uses machine learning to analyze resumes, categorizing them based on content. The application is designed to handle PDF files, which are processed to extract text and then classified.

### Prerequisites

Python (3.6 or higher recommended)
Required Python Libraries: numpy, pandas, nltk, scikit-learn, pdfminer.six
A dataset of resumes in a CSV format

### Warning:
If you are connected to a Jio/campus/office network, it may have blacklisted the git repository that nltk package refers to. The developers suggest a personal preference of Airtel internet because they only had that alternative internet.

### Use Instructions

1. To use our Resume_Screening application, you need to first save your resume to the folder resumes which can be navigated from root directory in the following manner:
    ## Resume_Screening > uAgents > python > src > resumes

2. While using the application, please enter the name of the resume as it is, without any difference and along with the ".pdf" extension name. 

### Installation Instructions

1. After opening the terminal with its current directory at root, navigate to Resume_Screening > uAgents > python by using the cd command in terminal.

2. Open the command line terminal (Windows) or terminal (Linux and Mac) and type and enter the following commands sequentially:
    ## poetry install
    ## poetry shell

3. After the above steps, naviagte to src > Agents from the current directory and type and enter the following commands:
    ## pip install -r requirements.txt

4. After the above steps, type and enter the following commands in the terminal:
    
    # For Windows:
    ## python client.py

    # For Mac and Ubuntu:
    ## python3 client.py

5. Good day, comrades! 

### Clone the Repository:

Clone the project repository to your local machine to run it.
 ## git clone https://github.com/swetanshusbp/HackAI_Hack-230550_Final_Round

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


