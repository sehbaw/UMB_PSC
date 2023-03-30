#!/usr/bin/env python
# coding: utf-8

# ctrl+shift+p for command palette

# In[ ]:


#quick sample
import pandas as pd

data = {'ID': ['1900', '1901', '1902', '1903', '1903'],
        'Drug': ["","",""]
        }

df = pd.DataFrame(data)

print(df)


# In[ ]:


#generate the alphanumeric ID

import random, string
#string.ascii_uppercase - if I wanted to do both
x = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(8))
print(x)


# n_points = 10000
# 
# points = [] #should be an array or a series? since we are using pandas
# for i in range(n_points):
# 

# In[ ]:


#quick sample
import pandas as pd
import numpy as np 

data = {#'ID': generate_id(),
        #'Drug': np.random.normal(0.4, 0.025, size=(0.2, 0.6)) 
        'Drug': np.random.normal(0.4, 0.025, size=None) 
        }

df = pd.DataFrame(data)

print(df)


# In[ ]:


#without the ASCII
import random, string
#for 
x = ''.join(random.choice(string.digits) for i in range(8))
print(x)


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# In[ ]:


import pandas as pd
import random
import random, string 
n_points = 10000

x = ''.join(random.choice(string.digits) for _ in range(8))
print(x)

# dictionary to store the data

#for n in range(10000):
    #would a list comprehension be better? 
# np.random.rand(n_points)
data = {
    'ID': np.random.rand(n_points),
    "Absorbance": 
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

#how do we want to inspect the data? head() isn't really applicable here


# In[ ]:





# # working code starts here. 

# In[ ]:


import pandas as pd
import numpy as np
import random, string


#id = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(8))
# create a range of sample IDs
#sample_ids = [f' ID {id}' for i in range(10000)] #this code doesnt work because it will not generate a new 

def generate_id():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(8))

# create a range of sample IDs
sample_ids = [f'ID {generate_id()}' for i in range(10000)]

#generate date
date = pd.date_range('2023-01-03', '2023-01-31') # over the span of a month 

#generate temp
temp = [np.random.uniform(21,24) for i in range(10000)] #these are decimals because that is the only way I could 

# generate in a range 
absorbances = np.random.normal(0.3, 0.6, size=10000)

target_absorbances = (absorbances >= 0.35) & (absorbances <= 0.54)
sample = np.array(sample_ids)[target_absorbances][:1000]

# create a dataframe
data = pd.DataFrame({'Sample ID': sample_ids,
                     'Temp': temp,
                     'Date': np.random.choice(date, size=10000), # had to add a size because other generated the same date
                     'Absorbance': np.random.choice(absorbances, size=10000)})

data.to_csv(index=False)


# In[ ]:


#filtering the target absorbance to make sure that there are at least 1000 data points..not sure if this is the best way to handle this though

target_absorbances = (absorbances >= 0.35) & (absorbances <= 0.54)
sample = np.array(sample_ids)[target_absorbances][:1000] #we want to use 


# In[ ]:


#plot the data 


# In[ ]:


#if we want to send csv to a zcop 
compression_opts = dict(method='zip',
                        archive_name='candidate.csv')  
data.to_csv('candidate.zip', index=False,
          compression=compression_opts) 


# # Dirtying the Data 

# In[ ]:


#dirtying int the data 
    #randomly insert null values 
        # set the probability of a cell being set to NA
            #need to 
        # loop through the DataFrame and set some values to NA
            for i in range(len(data)):
                for col in data.columns:
                    if random.random() < probability:
                        df.at[i, col] = np.nan #.loc doesn't really apply here 


# How can we dirty the data? 
# - Null values randomnly inserted or replaced
# - outliers can be found in initial data but also could implement our own
# - throw in spaces...not necessarily null values but just break upthe data kinda with random cells 
# - can combine cells with two points together seperated by a comma and they have to break those
# - trailing spaces and white spaces
# - maybe have to perform data transformation
# 

# # visualization - mix of pyplot and seaborn 
# 

# In[ ]:


#https://seaborn.pydata.org/examples/layered_bivariate_plot.html 

#relplot obviously 

#i have more drawings of this than code so I am gonna work on making the drawings into code. 


