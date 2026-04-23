import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("ML\\student_data.csv")

veriler=df.head()
# print(veriler)
# print(df.info()) 
# print(df.isnull().sum())
# print(df.describe())

df = pd.get_dummies(df, drop_first=True) # drop_first=True, kukla değişken tuzağını önler.

x=df.drop("G3", axis=1)
# y=df["G3"]

y=(df["G3"]>=10).astype(int)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=2000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
# print(f"Model Dogruluk Skoru: {accuracy}")

# print(classification_report(y_test,y_pred))

results=X_test.copy()
results["Gercek_Deger"] = y_test
results["Tahmin_Deger"] = y_pred

riskli_ogrenciler=results[results["Tahmin_Deger"]==0]
print(f"Riskli Öğrenciler: {len(riskli_ogrenciler)}")
# print(riskli_ogrenciler[['age','absences','studytime', 'failures']].head(10))

#karar agaci modeli
tree_model=DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train,y_train)
tree_y_pred=tree_model.predict(X_test)
tree_accuracy=accuracy_score(y_test,tree_y_pred)
# print(f"Karar Agaci Dogruluk Skoru: {tree_accuracy}")
# print(classification_report(y_test,tree_y_pred))

importances=pd.Series(tree_model.feature_importances_,index=x.columns)

importances.nlargest(10).plot(kind='barh')
plt.title("Model icin en onemli 10 ozellik")
# plt.show()


