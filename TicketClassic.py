#imported libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


nltk.download("stopwords")
#Load The Data set
df=pd.read_csv("customer_support_tickets.csv")
stop_words=set(stopwords.words("english"))

#Apply TF-IDF
def clean_text(text):
    text=str(text).lower()
    text=re.sub(r"[^\w\s]","",text)
    words=text.split()
    words=[word for word in words if word not in stop_words]
    return " ".join(words)
df["clean_text"]=df["Ticket Description"].apply(clean_text)
vectorizer=TfidfVectorizer()
#Split The data
X=vectorizer.fit_transform(df["clean_text"])
Y=df["Ticket Type"]
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

#Train The Ticket Type
model=LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)
predictions=model.predict(X_test)
print(predictions[:5])

#Calculate accuracy
accuracy=accuracy_score(Y_test,predictions)
print("Category Prediction Accuracy:",accuracy)
print(classification_report(Y_test,predictions))
Y_priority=df["Ticket Priority"]
X_train,X_test,Y_priority_train,Y_priority_test=train_test_split(
    X,
    Y_priority,
    test_size=0.2,
    random_state=42
)


#Train The Ticket priority
model2=LogisticRegression(max_iter=1000)
model2.fit(X_train,Y_priority_train)
predictions1=model2.predict(X_test)


#Calculate accuracy
accuracy2=accuracy_score(Y_priority_test,predictions1)
print("Priority Prediction Accuracy:",accuracy2)
print(classification_report(Y_priority_test,predictions1))

#Accept User Input
New_Description=input("Enter Your Problem: ")
clean_Description=clean_text(New_Description)
Vector_Description=vectorizer.transform([clean_Description])
Actual_Type_prediction=model.predict(Vector_Description)
Actual_Priority_prediction=model2.predict(Vector_Description)

#Predict The result
print("Predicted Category:",Actual_Type_prediction[0])
print("Predicted Priority:",Actual_Priority_prediction[0])




