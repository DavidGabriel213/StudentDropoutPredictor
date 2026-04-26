from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import joblib
import os
processor=joblib.load("Preprocessors/PREPROCESSOR.joblib")
model=joblib.load('models/XGBMODEL.joblib')
print('processor and model loaded')
app=Flask(__name__)
@app.route('/', methods=['GET','POST'])
def myFunc():
    prediction=None
    result_class=None 
    if request.method=='POST':
        State=request.form['state']
        Institution=request.form['institution']
        Level=request.form['level']
        Course=request.form['course']
        ParentEducation=request.form['parentedu']
        TuitionPaid=request.form['tuition_paid']
        
        Age=float(request.form['age'])
        Cgpa=float(request.form['cgpa'])
        AttendanceRate=float(request.form['attendance_rate'])
        DistanceFromCampus=float(request.form['distance'])
        DailyStudyHours=float(request.form['study_hours'])
        FailedCourses=float(request.form['failed_courses'])
        MentalHealthScore=float(request.form['health_score'])
        ExtracurricularActivities=float(request.form['extra_activities']) 
        LecturerRating=float(request.form['rating'])
        LibraryVisitsPerMonth=float(request.form['library_visits'])
        ExamsScore=float(request.form['exams_score'])
        FamilyIncome=float(request.form['family_income'])
        
        Gender=float(request.form['gender'])
        Scholarship=float(request.form['scholarship'])
        InternetAccess=float(request.form['internetaccess'])
        PartTimeJob=float(request.form['partjob'])
        HostelResident=float(request.form['hostelresident'])
        PrevDropoutAttempt=float(request.form['prev_drop_attempt'])
        
        FamilyIncome_log=np.log1p(FamilyIncome)
        
        features=pd.DataFrame({
            'State':[State], 'Institution':[Institution],'Level':[Level],'Course':[Course], 'ParentEducation':[ParentEducation],
            'TuitionPaid':[TuitionPaid],
            'Age':[Age], 'CGPA':[Cgpa], 'AttendanceRate(%)':[AttendanceRate],
            'DistanceFromCampus(km)':[DistanceFromCampus],'DailyStudyHours':[DailyStudyHours],
            'FailedCourses':[FailedCourses], 'MentalHealthScore':[MentalHealthScore],
            'ExtracurricularActivities':[ExtracurricularActivities], 'LecturerRating':[LecturerRating],
            'LibraryVisitsPerMonth':[LibraryVisitsPerMonth], 'ExamScoreAverage(%)':[ExamsScore],
            'FamilyIncome_log':[FamilyIncome_log],
            'Gender':[Gender], 'Scholarship':[Scholarship], 'InternetAccess':[InternetAccess],
            'PartTimeJob':[PartTimeJob], 'HostelResident':[HostelResident], 'PrevDropoutAttempt':[PrevDropoutAttempt]
        })
        Processed_features=processor.transform(features) 
        result=model.predict(Processed_features)[0]
        if result==0:
            prediction='At-risk'
        elif result==1:
            prediction='Dropped-out'
        elif result==2:
            prediction='Graduated'
        else:
            prediction='Suspended'
        class_map={
            'Graduated':'result-graduated',
            'At-risk': 'result-atrisk',
            'Suspended': 'result-suspended',
            'Dropped-out':'result-dropout'
        }  
        result_class=class_map.get(prediction,'')
                                               
    return render_template('drop_design.html',prediction=prediction,result_class=result_class)
if __name__==('__main__'):
    port =int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=True)
