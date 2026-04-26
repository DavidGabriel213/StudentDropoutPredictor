import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/USER/Desktop/DropoutPredictor/Data/NigerianDropout.csv")

df['StudentID']=df['StudentID'].astype(str).str.strip()
df['Age']=np.abs(df['Age'].astype(int))

df['Gender']=df['Gender'].astype(str).str.capitalize().str.strip()
gender_corrector={'M':'Male','F':'Female'}
df['Gender']=df['Gender'].replace(gender_corrector)

print(df['State'].isnull().sum())
df['State']=df['State'].astype(str).str.capitalize().str.strip()

df['Institution']=df['Institution'].astype(str).str.capitalize().str.strip()
print(df['Institution'].value_counts())
institution_corrector={'Poly':'Polytechnic','Coe':'College of education'}
df["Institution"]=df['Institution'].replace(institution_corrector)

df['Level']=df['Level'].astype(str).str.strip()
df['Level']=df['Level'].str.replace('Year','').str.strip()
def level_corrector(c):
    c=str(c)
    if len(c)==1:
        return str(int(c)*100)
    else:
        return c
df['Level']=df['Level'].apply(lambda x : level_corrector(x))
def level_corrector1(c):
    c=str(c)
    if 'L' not in c:
        return c + 'L'
    else:
        return c 
df['Level']=df['Level'].apply(lambda x : level_corrector1(x))
df['Level']=df['Level'].apply(lambda x : np.nan if x=='nanL' else x)
df['Level']=df['Level'].fillna(df.groupby('Age')['Level'].transform(lambda x: x.mode()[0]))

df['Course']=df['Course'].astype(str).str.capitalize().str.strip()
df['Course']=df['Course'].apply(lambda x: np.nan if x=='nan' else x)

df['CGPA']=df['CGPA'].astype(str).str.replace('/5.0','').str.strip()
df['CGPA']=np.abs(pd.to_numeric(df['CGPA'], errors='coerce'))
df['CGPA']=df['CGPA'].apply(lambda x : np.nan if (x<=0 or x>=5) else x)
df['CGPA']=(df['CGPA'].fillna(df.groupby(['Institution','Course','Gender'])['CGPA'].transform('mean'))).round(2)

df['AttendanceRate(%)']=df['AttendanceRate(%)'].astype(str).str.replace('%','').str.replace('-','').str.replace('percent','').str.strip()
df['AttendanceRate(%)']=pd.to_numeric(df['AttendanceRate(%)'],errors='coerce')
df['AttendanceRate(%)']=df['AttendanceRate(%)'].apply(lambda x : np.nan if (x<0 or x>100) else x)
df['AttendanceRate(%)']=(df['AttendanceRate(%)'].fillna(df.groupby(['Gender','Course','Level'])['AttendanceRate(%)'].transform('mean'))).round(1)

df['TuitionPaid']=df['TuitionPaid'].astype(str).str.capitalize().str.strip()
print(df['TuitionPaid'].value_counts())
tuition_corrector={'No':'Not paid','Yes':'Fully paid','Half':'Partial','0.5':'Partial','0':'Not paid','1':'Fully paid'}
df['TuitionPaid']=df['TuitionPaid'].replace(tuition_corrector)

df['Scholarship']=df['Scholarship'].astype(str).str.capitalize().str.strip()
scholarship_corrector={'0':'No','1':'Yes','N':'No','Y':'Yes'}
df['Scholarship']=df['Scholarship'].replace(scholarship_corrector)

df['FamilyIncome(NGN)']=df['FamilyIncome(NGN)'].astype(str).str.replace('-','').replace(',','').str.replace('"','').replace('NGN','').str.replace('\u20A6','').str.strip()
df['FamilyIncome(NGN)']=pd.to_numeric(df['FamilyIncome(NGN)'],errors='coerce')
max1=df['FamilyIncome(NGN)'].quantile(0.75)+1.5*(df['FamilyIncome(NGN)'].quantile(0.75)-df['FamilyIncome(NGN)'].quantile(0.25))
min1=df['FamilyIncome(NGN)'].quantile(0.25)-1.5*(df['FamilyIncome(NGN)'].quantile(0.75)-df['FamilyIncome(NGN)'].quantile(0.25))
df['FamilyIncome(NGN)']=df['FamilyIncome(NGN)'].apply(lambda x: np.nan if (x> max1 or x<min1) else x)
df['FamilyIncome(NGN)']=(df['FamilyIncome(NGN)'].fillna(df.groupby('TuitionPaid')['FamilyIncome(NGN)'].transform('mean'))).round(2)

df['DistanceFromCampus(km)']=df['DistanceFromCampus(km)'].astype(str).str.replace('km','').str.replace('-','').str.strip()
def distance_corrector(c):
    c=str(c)
    if 'miles' in c:
        k=float(c.replace('miles','').strip())*1.609
        return k
    else:
        return c
df['DistanceFromCampus(km)']=df['DistanceFromCampus(km)'].apply(lambda x: distance_corrector(x))
df['DistanceFromCampus(km)']=pd.to_numeric(df['DistanceFromCampus(km)'],errors='coerce')
df['DistanceFromCampus(km)']=(df['DistanceFromCampus(km)'].fillna(df.groupby('Institution')['DistanceFromCampus(km)'].transform('mean'))).round(1)

df['DailyStudyHours']=df['DailyStudyHours'].astype(str).str.replace('hours','').str.replace('hrs','').str.replace('-','')
df['DailyStudyHours']=pd.to_numeric(df['DailyStudyHours'],errors='coerce')
df['DailyStudyHours']=(df['DailyStudyHours'].fillna(df.groupby(['Course','Scholarship'])['DailyStudyHours'].transform('mean'))).round(1)

df['FailedCourses']=df['FailedCourses'].apply(lambda x: np.nan if x>6 else x)
df['FailedCourses']=df['FailedCourses'].fillna(df.groupby(['Course','Institution','Level'])['FailedCourses'].transform(lambda x: x.mode()[0]))
df['FailedCourses']=df['FailedCourses'].astype(int)

df['PartTimeJob']=df['PartTimeJob'].astype(str).str.capitalize().str.strip()
job_corrector={'0':'No','1':'Yes','N':'No','Y':'Yes'}
df['PartTimeJob']=df['PartTimeJob'].replace(job_corrector)

df['MentalHealthScore']=df['MentalHealthScore'].astype(str).str.replace('out of 10','').str.replace('-','').str.replace('/5.0','')
df['MentalHealthScore']=pd.to_numeric(df['MentalHealthScore'],errors='coerce')
df['MentalHealthScore']=df['MentalHealthScore'].apply(lambda x: x/10 if x>10 else x)
df['MentalHealthScore']=(df['MentalHealthScore'].fillna(df.groupby(['Gender','Course'])['MentalHealthScore'].transform('mean'))).round(1)

df['InternetAccess']=df['InternetAccess'].astype(str).str.capitalize().str.strip()
internet_corrector={'Limited':'No','1':'Yes','0':'No'}
df['InternetAccess']=df['InternetAccess'].replace(internet_corrector)

df['ParentEducation']=df['ParentEducation'].astype(str).str.capitalize()
df['ParentEducation']=df['ParentEducation'].apply(lambda x: np.nan if x=='Nan' else x)
df['ParentEducation']=df['ParentEducation'].fillna(df.groupby('State')['ParentEducation'].transform(lambda x : x.mode()[0]))
print(df['ParentEducation'].isnull().sum())

df['HostelResident']=df['HostelResident'].astype(str).str.capitalize()
hostel_corrector={'0':'No','1':'Yes'}
df['HostelResident']=df['HostelResident'].replace(hostel_corrector)

df['LecturerRating']=df['LecturerRating'].astype(str).str.replace('stars','').str.replace('/5','').str.strip()
df['LecturerRating']=pd.to_numeric(df['LecturerRating'],errors='coerce')
df['LecturerRating']=df['LecturerRating'].apply(lambda x: x/10 if x>10 else x)
df['LecturerRating']=(df['LecturerRating'].fillna(df.groupby('FailedCourses')['LecturerRating'].transform('mean'))).round(1)

df['ExamScoreAverage(%)']=df['ExamScoreAverage(%)'].astype(str).str.replace('/100','').str.replace('marks','').str.replace('-','').str.strip()
df['ExamScoreAverage(%)']=pd.to_numeric(df['ExamScoreAverage(%)'],errors='coerce')
df['ExamScoreAverage(%)']=(df['ExamScoreAverage(%)'].apply(lambda x: x/10 if x>100 else x)).round(1)
df['ExamScoreAverage(%)']=df['ExamScoreAverage(%)'].fillna(df.groupby('Course')['ExamScoreAverage(%)'].transform('mean'))
df['ExamScoreAverage(%)']=df['ExamScoreAverage(%)'].round(2)

df['StudentStatus']=df['StudentStatus'].astype(str).str.capitalize().str.strip()
status_corrector={'G':'Graduated','Completed':'Graduated','Pass':'Graduated','Left':'Dropped-out',
                  'D':'Dropped-out','Do':'Dropped-out',
                  'Dropout':'Dropped-out','Dropped':'Dropped-out',
                  'Ar':'At-risk','A':'At-risk','A':'At-risk','Struggling':'At-risk','Atrisk':'At-risk',
                  'Deferred':'Suspended','On hold':'Suspended','S':'Suspended',
                  'Nan':np.nan}
df['StudentStatus']=df['StudentStatus'].replace(status_corrector)

df['FamilyIncome_log']=np.log1p(df['FamilyIncome(NGN)']).round(4)
df['CGPA_index']=(df['DailyStudyHours']/df['CGPA']).round(3)

df.to_csv('cleaned&engineering.csv', index=False)



