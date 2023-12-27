#for the agents
import os
from uagents import Agent, Bureau, Context, Model

#for the models
import numpy as np
import pandas as pd
import re
import nltk
import base64
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from pdfminer.high_level import extract_text

base_path = "../resumes/"
model_path = "../model/screening.py"





# In[2]:


# Ensure nltk resources are downloaded (do this once)
nltk.download('stopwords')


# In[3]:


# Data cleaning function
def cleanResume(resumeText):
    resumeText = re.sub('http\\S+\\s*', ' ', resumeText)  # Remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # Remove RT and cc
    resumeText = re.sub('#\\S+', '', resumeText)  # Remove hashtags
    resumeText = re.sub('@\\S+', '  ', resumeText)  # Remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', resumeText)  # Remove punctuations
    resumeText = re.sub(r'[^\\x00-\\x7f]', r' ', resumeText)  # Remove non-ASCII characters
    resumeText = re.sub('\\s+', ' ', resumeText)  # Remove extra whitespace
    return resumeText


# In[4]:


# Load dataset
resume_dataset_path = "UpdatedResumeDataSet.csv"  # Replace with your dataset path
resume_dataset = pd.read_csv(resume_dataset_path, encoding='utf-8')
resume_dataset['cleaned_resume'] = resume_dataset['Resume'].apply(cleanResume)


# In[5]:


# Encoding categorical data
label_encoder = LabelEncoder()
resume_dataset['Category'] = label_encoder.fit_transform(resume_dataset['Category'])


# In[6]:


# Feature extraction
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
X = tfidf_vectorizer.fit_transform(resume_dataset['cleaned_resume'])
y = resume_dataset['Category']


# In[7]:


# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Model training
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)


# In[9]:


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


# In[10]:


# Function to predict category
def predict_category(new_resume, model, vectorizer, label_encoder):
    cleaned_resume = cleanResume(new_resume)
    features = vectorizer.transform([cleaned_resume])
    prediction = model.predict(features)
    predicted_category = label_encoder.inverse_transform(prediction)
    return predicted_category[0]


# In[11]:


# Function to predict category of a resume in PDF format
def predict_category_pdf(pdf_path, model, vectorizer, label_encoder):
    resume_text = extract_text_from_pdf(pdf_path)
    return predict_category(resume_text, model, vectorizer, label_encoder)


# In[12]:


# Function to decode base64 and save as PDF
def save_base64_to_pdf(base64_string, output_path):
    pdf_data = base64.b64decode(base64_string)
    with open(output_path, 'wb') as file:
        file.write(pdf_data)
    return output_path




#Do not run this while on campus/college wifi or on Jio network

def screening(path):
    predicted_category = predict_category_pdf(path, clf, tfidf_vectorizer, label_encoder)
    return predicted_category
class Message(Model):
    message: str

client = Agent(name = "client", seed = "client path")
model =  Agent(name = "client", seed = "model")

@client.on_event("startup")
async def get_file(ctx: Context):
    ctx.logger.info(f'Please enter the name of the new resume along with .pdf extension in the resumes folder to be assessed')
    n = input()
    pdf_path = os.path.join(base_path,n)
    ctx.logger.info(f'"{pdf_path}" sent as path of the resume to be reviewed')
    await ctx.send(model.address,Message(message=pdf_path))
    
@model.on_message(model=Message)
async def get_result(ctx:Context, sender:str, msg:Message):
    result = screening(msg.message)
    ctx.logger.info(f'This applicant is favourable for the position of: {result}')
    
bureau = Bureau()
bureau.add(client)
bureau.add(model)
if __name__ == "__main__":
    bureau.run()
    


    
    
