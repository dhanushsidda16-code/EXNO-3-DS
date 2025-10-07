## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
 import pandas as pd 
     df= pd.read_csv("/content/Encoding Data.csv")
     df
```
<img width="554" height="458" alt="Screenshot 2025-10-07 155649" src="https://github.com/user-attachments/assets/bc3bf146-4d91-4a46-895e-077834b05de1" /> 

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm= ['Hot','Warm','Cold']
e1= OrdinalEncoder (categories=[pm])
e1.fit_transform (df[["ord_2"]])
```
<img width="297" height="245" alt="Screenshot 2025-10-07 160555" src="https://github.com/user-attachments/assets/aed14bca-9baf-40f5-9b36-062b7d07098e" /> 

```
df['bo2']= e1.fit_transform(df[["ord_2"]])
df
```
<img width="589" height="450" alt="Screenshot 2025-10-07 160647" src="https://github.com/user-attachments/assets/bd49f4e7-3dc6-4001-80d4-db016115b719" /> 

```
le= LabelEncoder()
dfc= df.copy()
dfc['ord_2']=le.fit_transform (dfc['ord_2'])
dfc
```
<img width="593" height="447" alt="Screenshot 2025-10-07 160737" src="https://github.com/user-attachments/assets/c7dd5978-2a0b-4d07-abc7-699b2fd3835f" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2= pd.concat([df2,enc],axis=1)
df2
```
<img width="570" height="445" alt="Screenshot 2025-10-07 160849" src="https://github.com/user-attachments/assets/5e7563a8-d20d-4bb9-bdf1-b87beb731677" /> 

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="843" height="442" alt="Screenshot 2025-10-07 160939" src="https://github.com/user-attachments/assets/9d5a80a8-3c24-4308-b06a-4f87d660cc50" />

```
from category_encoders import BinaryEncoder
df= pd.read_csv("/content/data.csv")
df
```
<img width="632" height="455" alt="Screenshot 2025-10-07 161032" src="https://github.com/user-attachments/assets/9f1d437e-a31c-4e56-a94a-827b2f16a45f" />

```
be= BinaryEncoder()
nd= be.fit_transform(df['Ord_2'])
df
```
<img width="632" height="453" alt="Screenshot 2025-10-07 161124" src="https://github.com/user-attachments/assets/862dc5ee-1414-400a-9b92-0de7a16e3c42" />

```
dfb= pd.concat([df,nd],axis=1)
dfb
```
<img width="889" height="452" alt="Screenshot 2025-10-07 161206" src="https://github.com/user-attachments/assets/8944bd68-4d9e-4372-8b33-37ab77c7c529" /> 

```
from category_encoders import TargetEncoder
te= TargetEncoder()
CC= df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC= pd.concat([CC,new],axis=1)
CC
```
<img width="717" height="454" alt="Screenshot 2025-10-07 161253" src="https://github.com/user-attachments/assets/d03eb0c3-82b5-428a-9a5e-b58031a39e64" />

```
import pandas as pd 
import numpy as np
from scipy import stats 
df= pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="1002" height="492" alt="Screenshot 2025-10-07 161346" src="https://github.com/user-attachments/assets/ec31ebd8-d810-47c0-b6bf-98104e2508d7" /> 

```
df.skew()
```
<img width="467" height="225" alt="Screenshot 2025-10-07 161452" src="https://github.com/user-attachments/assets/3ba35dc9-05c1-4850-a9f2-92fca6185017" /> 

```
np.log(df["Highly Positive Skew"])
```
<img width="495" height="562" alt="Screenshot 2025-10-07 161543" src="https://github.com/user-attachments/assets/a77e3311-e664-454e-9942-d3b2232b4837" /> 

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="435" height="577" alt="Screenshot 2025-10-07 161621" src="https://github.com/user-attachments/assets/e278505b-bf56-4ba2-b4cb-69cfcdc9b30d" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="447" height="571" alt="Screenshot 2025-10-07 161658" src="https://github.com/user-attachments/assets/5e33c918-7b76-4e94-ab7b-e24502651901" />

```
np.square(df["Highly Positive Skew"])
```
<img width="470" height="566" alt="Screenshot 2025-10-07 161734" src="https://github.com/user-attachments/assets/94a60dca-c339-4094-ac29-11025e58a2c0" />

```
df["Highly Positive Skew_boxcox"],parameters= stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1280" height="519" alt="Screenshot 2025-10-07 161818" src="https://github.com/user-attachments/assets/994ba38c-6e0b-46a2-8c25-b3d2fc80cab5" />

```
df.skew()
```
<img width="483" height="310" alt="Screenshot 2025-10-07 161900" src="https://github.com/user-attachments/assets/9f0390b9-41eb-4c89-bb88-f82c4b65355f" />

```
df["Highly Negative Skew_yeojohnson"],parameters= stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="507" height="336" alt="Screenshot 2025-10-07 161939" src="https://github.com/user-attachments/assets/31fcde01-bc3d-4dd7-9e9a-30b33c09eee8" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1724" height="543" alt="Screenshot 2025-10-07 162030" src="https://github.com/user-attachments/assets/7a8eb7ce-e015-48bc-aef8-d53310a22f93" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="912" height="558" alt="Screenshot 2025-10-07 162110" src="https://github.com/user-attachments/assets/9edf55a3-163a-453f-bf16-67bd8bb92245" /> 

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="877" height="551" alt="Screenshot 2025-10-07 162156" src="https://github.com/user-attachments/assets/280db443-fdba-4b22-a09e-d500754e0652" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="954" height="558" alt="Screenshot 2025-10-07 162230" src="https://github.com/user-attachments/assets/94fa3a7f-b912-4bb1-9ef6-ca752656b76a" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
<img width="804" height="563" alt="Screenshot 2025-10-07 162305" src="https://github.com/user-attachments/assets/e1798f47-3bdd-4604-a410-d4d6b5ca4614" />

```
dt =pd.read_csv("titanic_dataset.csv")
dt
dt=pd.read_csv("titanic_dataset.csv")
dt
```
<img width="1684" height="522" alt="Screenshot 2025-10-07 162358" src="https://github.com/user-attachments/assets/ce13c222-9089-4d9d-965c-d806f4ea1ba4" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="886" height="552" alt="Screenshot 2025-10-07 162435" src="https://github.com/user-attachments/assets/4b1190fd-89dd-44dd-9775-db196a8f3bac" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="866" height="556" alt="Screenshot 2025-10-07 162518" src="https://github.com/user-attachments/assets/b04b7eb4-dc81-43ae-ae4f-8ee28082ad3b" />

# RESULT:
      We have successfully executed feature encoding and transformation process using the given dataset
       
