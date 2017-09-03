def predict_rent():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib
    import csv
    import matplotlib.pyplot as plt
    from scipy.stats import skew
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.preprocessing import Imputer
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LassoCV

    #Importing Data
    fileloc = "https://ndownloader.figshare.com/files/7586326"
    df = pd.read_csv(fileloc)
    Y=df.uf17
    
    #Deleting few rows (More explaination on logic on Readme.md)
    delrows = pd.read_csv('rows.csv')
    dr = []
    for row in delrows.iterrows():
        dr.append(row[1][0])
    df = df.drop(df.index[dr], axis =0)
    Y = Y.drop(Y.index[dr], axis = 0)
    
    #Removing rows with outlier in response variable Y=99999
    keep_rows=Y!=99999
    Y=Y.loc[keep_rows]
    df=df.loc[keep_rows]
    
    
    #Diving relevant features into caterical and continuous sets
    cate = ['boro','sc36','sc37','sc38','sc54','sc114','sc115','sc116','sc117','sc120','sc121','sc127','sc141','sc143',
            'sc144','sc23','sc147','sc149','sc173','sc171','sc152','sc153','sc154','sc155','sc156','sc157','sc158',
            'sc159','sc161','sc164','sc166','sc174','sc181','sc541','sc184','sc542','sc543','sc544','sc185','sc189',
            'sc197','sc198','sc187','sc188','sc199','sc190','sc191','sc192','sc193','sc194','sc548','sc549','sc550',
            'sc551','sc575','uf19','new_csr','rec15','sc26','rec21','rec1','rec4','rec62','rec64','rec54','rec53','cd',
            'uf1_1','uf1_2','uf1_3','uf1_4','uf1_7','uf1_8','uf1_9','uf1_12','uf1_13','uf1_14','uf1_15','uf1_17',
            'uf1_18','uf1_19','uf1_20','sc27','uf1_5','uf1_6','uf1_10','uf1_11','uf1_16','uf1_21','uf1_22','uf1_35',
            'sc24','sc118','hflag6','hflag13','hflag1','hflag3','hflag14','hflag7','hflag9','hflag10','hflag91',
            'hflag11','hflag12']
    
    conti = ['uf5','uf6','uf7','uf7a','uf9','uf8','uf10','sc150','sc151','uf12','uf13','uf14','uf15','uf16','uf64',
             'uf17','sc196','uf23','uf34','uf35','uf36','uf37','uf38','uf39','uf40','uf48','uf11','sc186','sc571']
    
    
    continuous_df = df[conti]
    
    #Performing feature engineering on continuous features 
    continuous_df1 = continuous_df.drop(['uf5','uf6','uf7','uf7a','uf9','uf8'],1)
    continuous_df['uf10']=continuous_df['uf10'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], 
                     [50,150,250,350,450,550,650,750,850,950,1125,1375,1625,1875,2250,2750,3250,3750,4250,4750,5250,5750,6250,6750,7250,7750,8250,8750,9500,10000]) 
    continuous_df['uf12']=continuous_df['uf12'].replace([9999],[98])
    continuous_df['uf64']=continuous_df['uf64'].replace([9998,9999],[98,0])
    continuous_df['sc196']=continuous_df['sc196'].replace([1,2,3,4,8],[4,3,2,1,98])
    continuous_df['uf23']=continuous_df['uf23'].replace([1,2,3,4,5,6,7,8,9,10],[2005,1995,1985,1976,1967,1953,1938,1925,1911,1890])
    continuous_df['uf48']=continuous_df['uf48'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13],[1,1,2,2,3,4,5,7.5,11,16,34.5,75,110])
    continuous_df['uf11']=continuous_df['uf11'].replace([1,2,3,4,5,6,7],[1.5,3,4,5,8,14.5,25])
    continuous_df['sc186']=continuous_df['sc186'].replace([2,3,4,5,8,9],[1,2,3,4,98,0])
    continuous_df['sc571']=continuous_df['sc571'].replace([1,2,3,4,5,8],[0,2.5,12.5,20,98,98])
    uf10_bool=(continuous_df.uf10==99)
    continuous_df.uf10_bool=uf10_bool
    continuous_df.uf10=continuous_df.uf10.replace(99,0)
    
    #Performing feature engineering on categorical features
    categorical_df = df[cate]
    categorical_df['sc166'] = categorical_df['sc166'].replace([1,2,3],[1,0,0])
    categorical_df['sc193'] = categorical_df['sc193'].replace([2,3,8,9],[2,3,0,0])
    categorical_df['sc114'] = categorical_df['sc114'].replace([1,2,3,4],[0,1,1,3])
    categorical_df['sc120'] = categorical_df['sc120'].replace([1,2,3,4,5,8],[1,2,2,4,5,np.nan])
    categorical_df['sc141'] = categorical_df['sc141'].replace([1,2,3,8,9],[1,0,0,np.nan,9])
    categorical_df['sc143'] = categorical_df['sc143'].replace([1,2,3,8,9],[1,2,3,np.nan,2])
    categorical_df['sc144'] = categorical_df['sc144'].replace([1,2,3,8,9],[1,0,0,np.nan,9])
    categorical_df['sc152'] = categorical_df['sc152'].replace([0,1,2],[0,1,1])
    categorical_df['sc154'] = categorical_df['sc154'].replace([1,2,3,8,9],[1,0,0,np.nan,9])
    categorical_df['sc155'] = categorical_df['sc155'].replace([0,1,2,3],[0,1,1,1])
    categorical_df['sc159'] = categorical_df['sc159'].replace([1,2,3],[1,0,0])
    categorical_df['sc161'] = categorical_df['sc161'].replace([1,2,3,9],[1,0,0,9])
    categorical_df['sc166'] = categorical_df['sc166'].replace([1,2,3],[1,0,0])
    categorical_df1 = categorical_df.drop(['sc117','rec1','rec4','sc24'],1) 
    
    #Combining both sets of features into one dataframe
    df_joint = pd.concat([continuous_df, categorical_df], axis=1)
    
    
    #A function to save NA values to treat them later using imputation
    def na_create(dataframe, file1):
        na_values = {}  
        with open(file1) as infile:
            reader = csv.reader(infile, delimiter=';')
            for row in reader:
                na_values[row[0]] = [int(i) for i in (row[1].split(','))]
        for column_name in dataframe.columns:
            if column_name in na_values.keys():
                series = dataframe[column_name]
                for val in na_values[series.name]:
                    series.replace(val, np.NaN, inplace=True)
                    dataframe[series.name] = series
        return dataframe
    
    #Saving the NA values dataframe to a csv file
    df_toImpute = na_create(df_joint, 'feature-na-values.csv')
    
    #Imputing categorical features and create dummy variables
    categorical_df = df_toImpute[cate]
    imputed = Imputer(np.NaN,strategy='most_frequent')
    cate_imputed = imputed.fit_transform(categorical_df)
    cate_imputed_df = pd.DataFrame(cate_imputed,columns=categorical_df.columns)
    cate_w_dummies = pd.get_dummies(data=cate_imputed_df, columns=categorical_df.columns)
    
    
    #Imputing continuous features
    conti_imputed=df_toImpute[conti]
    impu = Imputer(np.NaN)
    conti_imputed_df = impu.fit_transform(conti_imputed)
    conti_w_impu = pd.DataFrame(conti_imputed_df,columns=continuous_df.columns)

    #A function to scale continuous variables
    def ScaleConti(X):
       scaler = preprocessing.StandardScaler().fit(X)
       X_scaled=scaler.transform(X)
       return X_scaled
    continuous_new_df= ScaleConti(conti_w_impu)
    conti_df_scaled = pd.DataFrame(continuous_new_df, columns=conti)
    
    #Combining categorical and continuous features after imputation and scaling to one dataframe
    final_df = pd.concat([conti_df_scaled, cate_w_dummies], axis=1)
    unrelated_columns=['uf13','uf14','uf15','uf16']
    df_final= final_df.drop(unrelated_columns, 1)
    
    #Dropping target variable to create X and Y 
    X=df_final.drop('uf17',1)
    Y
    
    #Splitting Dataset into train and test
    X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=42)
    Xtest_array, ytest_array, ypred_array= model_fit(X_train,X_test,y_train,y_test)
    return Xtest_array, ytest_array, ypred_array

def model_fit(X_train,X_test,y_train,y_test):
    from sklearn.linear_model import LassoCV
    import numpy as np
    lasso_model = LassoCV()
    global lasso_fit
    lasso_fit = lasso_model.fit(X_train, y_train)
    y_pred = lasso_fit.predict(X_test)
    Xtest_array = np.array(X_test)
    ytest_array = np.array(y_test)
    ypred_array = np.array(y_pred)
    return Xtest_array, ytest_array, ypred_array

def score_rent():
    X_test, y_test, y_pred = predict_rent()
    R2 = lasso_fit.score(X_test,y_test)
    return R2

score_rent()
predict_rent()