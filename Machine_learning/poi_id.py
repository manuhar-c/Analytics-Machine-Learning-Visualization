
# coding: utf-8

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from time import time
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, chi2

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### All features were selected at this step for further analysis

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',

                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',

                 'expenses', 'exercised_stock_options', 'long_term_incentive',

                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',

                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Data Cleaning & Remove outliers
#defining a generic plot function for visual interpretation

def s_plot(x_feature, y_feature):
    features = ['poi', x_feature, y_feature]
    data = featureFormat(data_dict, features)
    import matplotlib.pyplot as plt
    for point in data:
        x = point[1]
        y = point[2]
        if point[0]:
            plt.scatter(x, y, color="r", marker=".")
        else:
            plt.scatter(x, y, color='g', marker=".")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.show()


### Data Exploration
names = data_dict.keys()
all_features = data_dict[names[0]]

print 'Number of datapoints: ',len(names)
print 'Number of features per datapoint: ',len(all_features)

### Finding POI's
poi_names=[]
for n in names:
    if data_dict[n]['poi'] == True:
        poi_names.append(n)
        
print 'Number of poi in datapoints: ',len(poi_names)
print 'Number of non-poi datapoints: ',len(names) - len(poi_names)

### Missing payments data
missing_payments=[]
for n in names:
    if data_dict[n]['total_payments'] == 'NaN' or data_dict[n]['total_payments']==0:
        missing_payments.append(n)
print 'Number of people with missing Total Payments Info: ',len(missing_payments)


#s_plot('salary','bonus')#visual plot, can see some outliers

#Removing NaN values & looking for irregular names
irregular_names=[]
for p in names:
    for feature in all_features:
        if data_dict[p][feature] == 'NaN'and feature != 'email_address': #Removing NaN except for email IDS
            data_dict[p][feature] = 0
         
    if len(p.split())not in [2,3]:# If name does not have 2 or 3 parts, then appending to a list for further checks
        irregular_names.append(p)
        
#print 'Irregular Names to check',' ',irregular_names

data_dict.pop('TOTAL') #Removing total
data_dict.pop('THE TRAVEL AGENCY IN THE PARK') #Removing travel agency

data_dict.pop('BANNANTINE JAMES M')#Removing outlier from Task 3, while creating new feature Stock Value to Salary

### Task 3: Create new feature(s)

people_keys = data_dict.keys()

#Creating features "to_poi_fraction","from poi fraction","shared poi receipt"

for person in people_keys:
    to_poi = float(data_dict[person]['from_this_person_to_poi'])
    from_poi = float(data_dict[person]['from_poi_to_this_person'])
    sh_rct_poi = float(data_dict[person]['shared_receipt_with_poi'])
    to_msg_total = float(data_dict[person]['to_messages'])
    from_msg_total = float(data_dict[person]['from_messages'])
    
    if from_msg_total > 0:
        data_dict[person]['to_poi_fraction'] = to_poi / from_msg_total
    else:
        data_dict[person]['to_poi_fraction'] = 0
        
    if to_msg_total > 0:
        data_dict[person]['from_poi_fraction'] = from_poi / to_msg_total
        data_dict[person]['shared_poi_rct'] = sh_rct_poi / to_msg_total
   
    else:
        data_dict[person]['from_poi_fraction'] = 0
        data_dict[person]['shared_poi_rct'] = 0

#Creating features "Salary_Fraction","Stock_to_salary"

    person_salary = float(data_dict[person]['salary'])
    person_tp = float(data_dict[person]['total_payments'])
    total_stock = float(data_dict[person]['total_stock_value'])
    
    if person_salary > 0 and person_tp > 0:
        data_dict[person]['salary_fraction'] = person_salary / person_tp
    else:
        data_dict[person]['salary_fraction'] = 0
    
    if person_salary > 0:  
        data_dict[person]['stock_to_salary'] = total_stock / person_salary
    else:
        data_dict[person]['stock_to_salary'] = 0
        
#Adding new features to feature_list        
features_list.extend(['to_poi_fraction','from_poi_fraction','salary_fraction','shared_poi_rct','stock_to_salary'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

###Creating pipeline with Feature Scaling, Feature selection, & Classifier
pipeline = Pipeline(steps=[('minmax_scaler', MinMaxScaler()), ('feature_selection', SelectKBest(chi2,k=9)), ('CF', DecisionTreeClassifier())])

###Defining Stratified shuffle split for validation
sss = StratifiedShuffleSplit(labels, 1000, random_state=42)

###Grid of potential classifiers
param_grid = [
        {    
        'CF':[GaussianNB()],  
        'feature_selection__k':[6,8,9,10]},
    
        {    
        'CF':[DecisionTreeClassifier()],
        'CF__min_samples_leaf':[1],
        'CF__min_samples_split':[2],
        'CF__max_depth':[1],
        'CF__class_weight':['balanced'],
        'feature_selection__k':[6,8,9,10]},
    
        {
        'CF':[AdaBoostClassifier()],
        'CF__base_estimator':[DecisionTreeClassifier(max_depth=1, min_samples_leaf=2,class_weight='balanced')],
        'CF__n_estimators':[50],
        'CF__learning_rate':[.8],
        'feature_selection__k':[6,8,9,10]},
    
        {    
        'CF':[ExtraTreesClassifier()],
        'CF__n_estimators':[10],
        'CF__min_samples_leaf':[1],
        'CF__min_samples_split':[2],
        'CF__max_depth':[None],
        'CF__class_weight':['balanced'],
        'feature_selection__k':[6,8,9,10]}]


###Defining gridsearch for selecting best classifier (Remove hashtag from line below to run the classifier search)
#clf = GridSearchCV(pipeline, param_grid,scoring=['recall','precision'],refit='recall',cv=sss)

###fitting to features & labels (Remove hashtag from line below to run the classifier GridSearchsearch)
#clf=clf.fit(features,labels)

###printing output from GridSearch (Remove hashtag from lines below to see the classifier GridSearchsearch output)
#best_params=clf.best_params_
#print best_params
#results=clf.best_score_
#print results



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### New parameters for parameter tuning
param_grid2={    
        'CF':[DecisionTreeClassifier()],
        'CF__min_samples_leaf':[1,2,3,4,5],
        'CF__min_samples_split':[2,3,4,5],
        'CF__max_depth':[1,2,3],
        'CF__class_weight':['balanced',None],
        'CF__criterion':['entropy','gini'],
        'CF__splitter':['best','random']}

### GridSearch for parameter tuning (Remove hashtag from lines below for classifier tuning GridSearchsearch)
#clf = GridSearchCV(pipeline, param_grid2,scoring=['recall','precision','f1'],refit='recall',cv=sss)
#clf=clf.fit(features,labels)

###printing output from GridSearch (Remove hashtag from lines below to see the classifier tuning GridSearchsearch output)
#best_params=clf.best_params_
#print best_params
#results=clf.best_score_
#print results

###Searching for best K value for SelectKBest for the tuned classifier with GridSearch
pipeline_ktune = Pipeline(steps=[('minmax_scaler', MinMaxScaler()), ('feature_selection', SelectKBest(chi2,k=16)),
                                 ('CF', DecisionTreeClassifier(min_samples_split=3,min_samples_leaf=1,
                                 splitter='best', class_weight = 'balanced', max_depth=2, criterion='gini'))])

param_grid_k={    
        'feature_selection__k':[2,4,6,8,10,12,14,16,18,20,22]}
        
###Remove hashtag below to run Gridsearch to find best k value for select K best for tuned classifier
#clf = GridSearchCV(pipeline_ktune, param_grid_k,scoring=['recall','precision'],refit='recall',cv=sss)
#clf=clf.fit(features,labels)

###printing output from GridSearch (Remove hashtag from lines below to see the classifier tuning GridSearchsearch output)
#best_params=clf.best_params_
#print best_params
#results=clf.best_score_
#print results

###Final classifier with tuned parameters based on GridSearch and Manual tuning
pipeline_final = Pipeline(steps=[('minmax_scaler', MinMaxScaler()), ('feature_selection', SelectKBest(chi2,k=16)),
                                 ('CF', DecisionTreeClassifier(min_samples_split=3,min_samples_leaf=1,
                                 splitter='best', class_weight = 'balanced', max_depth=2, criterion='gini'))])
clf=pipeline_final

### Final test with the tester.py code
print test_classifier(clf, my_dataset, features_list)


### Feature scores of final model
try:
    step = clf.named_steps['feature_selection']
    feature_scores = ['%.2f' % elem for elem in step.scores_ ]
    features_selected=[(features_list[i+1], feature_scores[i]) for i in step.get_support(indices=True)]
    features_selected = sorted(features_selected, key=lambda x: float(x[1]) , reverse=True)
    print 'Final Selected Features :',features_selected
    
except:
    pass

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
