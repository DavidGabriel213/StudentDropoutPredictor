import pandas as pd
import numpy as np
import matplotlib as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
import joblib

df=pd.read_csv('C:/Users/USER/Desktop/DropoutPredictor/NotebookS/Clin.csv')
df=df.drop(columns=['StudentID'])
# missing correction
df['ParentEducation']=df['ParentEducation'].fillna(df['ParentEducation'].mode()[0])
 
le=LabelEncoder()
df['StudentStatus']=le.fit_transform(df['StudentStatus'])
df['Scholarship']=le.fit_transform(df['Scholarship'])
df['InternetAccess']=le.fit_transform(df['InternetAccess'])
df['PartTimeJob']=le.fit_transform(df['PartTimeJob'])
df['HostelResident']=le.fit_transform(df['HostelResident'])
df['Gender']=le.fit_transform(df['Gender'])

cat_cols=['State','Institution','Level','Course','ParentEducation',
          'TuitionPaid']
bina_cols=[ 'Gender','Scholarship','InternetAccess','PartTimeJob',
           'HostelResident', 'PrevDropoutAttempt']
num_cols=['Age','CGPA','AttendanceRate(%)','DistanceFromCampus(km)',
          'DailyStudyHours','FailedCourses','MentalHealthScore',
          'ExtracurricularActivities','LecturerRating','LibraryVisitsPerMonth',
          'ExamScoreAverage(%)','FamilyIncome_log']
X=df[cat_cols + num_cols + bina_cols ]
y=df['StudentStatus']
#pipeline
processor=ColumnTransformer(transformers=[('ohe',
                                           OneHotEncoder(drop='first',
                                                         sparse_output=False,
                                                         handle_unknown='ignore'), cat_cols),
                                          ('scaler', StandardScaler(), num_cols)
                                          ],
                            remainder='passthrough')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_train_processed=processor.fit_transform(X_train)
X_test_processed=processor.transform(X_test)
joblib.dump(processor,'Preprocessor.joblib')

model=XGBClassifier(n_estimators=300,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsubsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=42,
                    n_jobs=-1
                    )

class_samples=compute_sample_weight('balanced',y_train)
model.fit(X_train_processed,y_train, sample_weight=class_samples)
y_pred_XGB=model.predict(X_test_processed)
accuracy=accuracy_score(y_test, y_pred_XGB)
report=classification_report(y_test,y_pred_XGB)
cm=confusion_matrix(y_test,y_pred_XGB)
Disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['t'])
print(f"XGBoost Acuracy: {accuracy:.4}")
print(f"report{report}")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
model1=DecisionTreeClassifier(max_depth=7)
model1.fit(X_test_processed,y_test)
pred=model1.predict(X_test_processed)
print(accuracy_score(y_test,pred))

model2=RandomForestClassifier(n_estimators=200,random_state=42)
model2.fit(X_train_processed,y_train)
pred1=model2.predict(X_test_processed)
print(accuracy_score(y_test,pred1))
print(classification_report(y_test,pred1))

model3=LogisticRegression()
model3.fit(X_train_processed,y_train)
pred2=model3.predict(X_test_processed)
print(accuracy_score(y_test,pred2))
print(classification_report(y_test,pred2))

from sklearn.model_selection import GridSearchCV
XGB_params={
    'n_estimators':[100,150,300],
    'learning_rate':[0.05,0.2,0.5,0.9],
    'subsample':[0.3,0.5,1.0],
    'max_depth':[3,7,9,None] 
     
}
xgb_tunned=GridSearchCV(
    XGBClassifier(use_label_encoder=False,
                  eval_metric='mlogloss',
                  random_state=42,
                  n_jobs=-1),
    XGB_params,
    cv=5,
    scoring='accuracy',
    verbose=1
)
xgb_tunned.fit(X_train_processed,y_train, sample_weight=class_samples)
print(xgb_tunned.best_params_)
print(xgb_tunned.best_estimator_)


