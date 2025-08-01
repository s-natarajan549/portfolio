import sklearn
from sklearn.model_selection import cross_val_score, KFold
import privacy_csv 
from privacy_csv import df 
from sklearn import preprocessing 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split 
import ast 
from sklearn import metrics
from sklearn.metrics import accuracy_score 


df_use = df[df['Category']!= '[]'].drop(0, axis = 0)
df_use['y'] = df_use['Category'].apply(lambda i: i[1:-1]) 
df_use['X'] = df_use['Text']
vectorizer = TfidfVectorizer(max_features = 10) #reduced for time's sake 
label_encoder = preprocessing.LabelEncoder() 

y_use = df_use['y']
y = label_encoder.fit_transform(y_use)

X_vectorized = vectorizer.fit_transform(df_use['X'])
X_gb = X_vectorized.toarray() 

naive_bayes = OneVsRestClassifier(MultinomialNB()) #multi-label naive bayes 


gb = GradientBoostingClassifier()
X_train, X_eval, y_train, y_eval = train_test_split(X_gb, y)
gb.fit(X_train, y_train)
y_preds = gb.predict(X_eval)

#calculate rmse 
accuracy = accuracy_score(y_train, y_preds)
print(accuracy)

"""

nb_scores = cross_val_score(naive_bayes, X_vectorized, y, cv=2, scoring = "accuracy")
print("Scores:", nb_scores)

gb_scores = cross_val_score(gb, X_gb, y, cv = 2, scoring = "accuracy")
print("GB Scores:", gb_scores)

"""






