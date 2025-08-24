#!/usr/bin/env python
# coding: utf-8

# # Rule Based Chatbot for college admission process using logistic regression
# 

# ## Importing libraries
# 

# In[5]:


import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# ## importing dataset

# In[7]:


df = pd.read_csv("chatbot_data.csv")
training_sentences = df["sentence"].tolist()
training_labels = df["label"].tolist()


# # Responses

# In[8]:


responses = {
    "greeting": ["Hello! Welcome to Department of Computer Application(Punjab University).", "Hi there! How can I help you?", "Greetings!"],
    "goodbye": ["Goodbye! Wish you the best.", "See you soon!", "Take care!"],
    "thanks": ["You're welcome!", "Glad to help!", "Anytime!"],
    "identity": ["I am  Department of Computer Application Admission ChatBot.", "You can call me Admission Assistant."],
    "courses": ["We offer MCA(Morning), MCA(Evening), M.Sc.(Data Science), Phd.", 
                "Available programs include Master's in Computer Application, M.S. in Data Science, Phd."],
    "admission_process": ["You can apply online at our college portal.", 
                          "Fill the admission form and submit required documents."],
    "documents": ["You need 10th, 12th, Graduation mark sheets, ID proof, and passport size photos.", 
                  "Submit academic certificates, transfer certificate, and ID proof."],
    "eligibility": ["Eligibility depends on the course. For UG: 12th pass. For PG: graduation required.", 
                    "You must meet the minimum marks criteria mentioned in the prospectus."],
    "last_date": ["The last date for admission is August 30.", "Admissions close by end of August."],
    "fees": ["The fee structure varies by course. Approx. 40,000 per year.", 
             "Tuition fee is mentioned on the website under 'Fee Structure'."],
    "hostel": ["Yes, hostel facilities are available for both boys and girls.", 
               "We provide hostel accommodation with mess facilities."],
    "location": ["Our college is located in Chandigarh, Punjab.", 
                 "Panjab University address: Sector 14, Chandigarh."],
    "entrance_exam": ["All courses require entrance exams.", 
                      "Entrance exam is required for all programs."]
}


# ## Vectorization (Convert text to numbers)

# In[9]:


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)


# ## Train Classifier

# In[10]:


clf = LogisticRegression(max_iter=200)
clf.fit(X_train, training_labels)


# ## Chat Loop

# In[11]:


print("Admission ChatBot \n Hello! Welcome to Department of Computer Application(Punjab University).\n Ask me about admission, courses, fees, or hostel. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Admission ChatBot: Goodbye! Wish you the best.")
        break

    X_test = vectorizer.transform([user_input.lower()])
    intent = clf.predict(X_test)[0]
    bot_response = random.choice(responses[intent])
    print("Admission ChatBot:", bot_response)
# In[12]:





# In[ ]:




