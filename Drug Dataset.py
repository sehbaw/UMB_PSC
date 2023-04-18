#!/usr/bin/env python
# coding: utf-8

# ctrl+shift+p for command palette

# In[ ]:


pwd


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


#https://towardsdatascience.com/what-are-quartiles-c3e117114cf1 

#import libraries 

#data

#create each quartile 


# In[ ]:





# # working code starts here. 

# In[4]:


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
absorbances = np.random.normal(0.15, 0.84, size=10000)


#target_absorbances = (absorbances >= 0.35) & (absorbances <= 0.54)

#sample = np.array(target_absorbances)[target_absorbances][:1000]

# create a dataframe
drug_candidate = pd.DataFrame({'Sample ID': sample_ids,
                     'Temp': temp,
                     'Date': np.random.choice(date, size=10000), # had to add a size because other generated the same date
                     'Absorbance': np.random.choice(absorbances, size=10000)})



#data.to_csv(index=False)

#print(target_absorbances)
#dirtying int the data 
    #randomly insert null values 
        # set the probability of a cell being set to NA
#def insert_null_data(dataset): #don't believe this needs a parameter
   # probability = 0.03 #want to keep the number small 
# loop through the DataFrame and set some values to NA
   # for i in range(len(drug_candidate)
      #  for col in drug_candidate.columns:
       #     if random.random() < probability:
          #       df.at[i, col] = np.nan #.loc doesn't really apply here 


#insert_null_data(drug_candidate)
# print(data)

#export to a txt 
#insert header
#with open("drug_candidate.txt", 'w', newline= '', encoding= 'utf8') as f:
 #   for item in drug_candidate: 
   #     f.write(item + '\n')
        
drug_candidate.to_csv('drug_candidate.txt', sep='\t', index=False, header=["Sample ID","Temp","Date","Absorbance"])


#df = drug_candidate.apply(insert_null_data(drug_candidate))
#print(df)


# In[ ]:


#testing..
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
absorbances = np.clip(np.random.normal(0.15, 0.84, size=10000), a_min = 0.15, a_max=0.9)

#quick to check to see if we are at 

for np.sum((absorbances >= 0.34) & (absorbances <= 0.54)) <=1000:
    #absorbances=np.clip(np.random.normal(0.21, 0.84, size=10000), a_min = 0.15, a_max=0.9)
    #absorbances = np.random.normal(0.21, 0.84, size=1000)
    
#print(absorbances)
    
    
#sample = np.array(target_absorbances)[target_absorbances][:1000]

# create a dataframe
#drug_candidate = pd.DataFrame({'Sample ID': sample_ids,
 #                    'Temp': temp,
  #                   'Date': np.random.choice(date, size=10000), # had to add a size because other generated the same date
   #                  'Absorbance': np.random.choice(absorbances, size=10000)})


drug_candidate = pd.DataFrame({'Sample ID': sample_ids,
                     'Temp': temp,
                     'Date': np.random.choice(date, size=10000), # had to add a size because other generated the same date
                     'Absorbance': np.random.choice(absorbances,  size=10000)})

#data.to_csv(index=False)


# print(data)

#export to a txt 
#insert header
#with open("drug_candidate.txt", 'w', newline= '', encoding= 'utf8') as f:
 #   for item in drug_candidate: 
   #     f.write(item + '\n')
        
drug_candidate.to_csv('drug_candidate.txt', sep='\t', index=False, header=["Sample ID","Temp","Date","Absorbance"])


    #NEED TO FIX THIS -- the insert null data function is doing this to me why 
#df = drug_candidate.apply(insert_null_data(drug_candidate))
#print(df)


# # Search and Set Data 
# 

# In[ ]:


#filtering the target absorbance to make sure that there are at least 1000 data points..not sure if this is the best way to handle this though

target_absorbances = (absorbances >= 0.35) & (absorbances <= 0.54) #this is essentially useless because doesn't respect the structure of the dataframe just is a conditional essentially 
sample = np.array(sample_ids)[target_absorbances][:1000] #we want to use 

drug_cand = drug_candidate

# ---  actual filter that is then sent to a txt file. 
filtered_df = drug_cand[(df["Absorbance"] >= 0.35) & (df['Absorbance'] <= 0.54)]
filtered_df.to_csv('filtered_df.txt', sep='\t', index=False, header=["Sample ID","Temp","Date","Absorbance"])


# display the filtered DataFrame


# In[ ]:





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
def insert_null_data(dataset): #don't believe this needs a parameter
    probability = 0.03 #want to keep the number small 
# loop through the DataFrame and set some values to NA
    for i in range(len(drug_candidate): 
         for col in drug_candidate.columns:
            if random.random() < probability:
                 df.at[i, col] = np.nan #.loc doesn't really apply here 


# How can we dirty the data? 
# - Null values randomnly inserted or replaced
# - outliers can be found in initial data but also could implement our own
# - throw in spaces...not necessarily null values but just break upthe data kinda with random cells 
# - can combine cells with two points together seperated by a comma and they have to break those
# - trailing spaces and white spaces
# - maybe have to perform data transformation -- why did I put this here. 
# 

# # visualization - mix of pyplot and seaborn 
# 

# In[ ]:


#https://seaborn.pydata.org/examples/layered_bivariate_plot.html 

#relplot obviously 

#i have more drawings of this than code so I am gonna work on making the drawings into code. 
# https://www.geeksforgeeks.org/how-to-set-axes-labels-limits-in-a-seaborn-plot/  -- set limits w/pyplot




# In[ ]:


#NOT using but just wanted to keep 
   #jointgrid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white", color_codes=True)

# Use JointGrid directly to draw a custom plot
g = sns.JointGrid(data=drug_candidate, x="Date", y="Absorbance", space=0, ratio=17)
g.plot_joint(sns.scatterplot, sizes=(30, 120),
            color="g", alpha=.6, legend=False, )
g.plot_marginals(sns.rugplot, height=1, color="g", alpha=.6)

g.set_ylim(0.15, 0.9)

#Axes.set_xlim(self, left=None, right=None, emit=True, auto=False, *, xmin=None, xmax=None)

#Axes.set_ylim(self, bottom=None, top=None, emit=True, auto=False, *, ymin=None, ymax=None)


# In[12]:


#unfiltered data 
import seaborn as sns 
import matplotlib.pyplot as plt
candidate = drug_candidate
sns.scatterplot(data=drug_candidate, x="Temp", y="Absorbance")
#took out size parameter
#sns.jointplot(data=drug_candidate[(drug_candidate["Absorbance"] >= 0.35) & (drug_candidate["Absorbance"] <= 0.54)], x="Temp", y="Absorbance", hue="Range", palette=["pastel"], marker="o", edgecolor="black", linewidth=0.5)

highlighted_data = candidate[(candidate["Absorbance"] >= 0.35) & (candidate["Absorbance"] <= 0.54)]
sns.scatterplot(data=highlighted_data, x="Temp", y="Absorbance", color="red", s=50, alpha=0.5) #was aiming for hue 
#legend set? 

#plt.axvspan(20, 30, color='gray', alpha=0.5)

#need to change axes limit -- set limit not working!!!

#Axes.set_ylim(left=0, right=0.9, emit=True, auto=False, xmin=None, xmax=None)
plt_ylim(0.1 , 1.01) #go beyond the the wanted limit 

#Graph Title -- 







# In[33]:


#lineplot -- needs to be edited
import seaborn as sns
sns.set_theme(style="ticks")

# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket_r")

# Plot the lines on two facets
sns.relplot(
    data=drug_candidate,
    x="time", y="firing_rate",
    hue="coherence", size="choice", col="align",
    kind="line", size_order=["T1", "T2"], palette=palette,
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)


# In[ ]:


#regression line
sns.lmplot(data=drug_candidate, x='Temp', y='Absorbance')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load a dataset
tips = sns.load_dataset("tips")

# Create a seaborn scatter plot
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", size="size")

# Highlight a range of dots
sns.scatterplot(data=tips[(tips["total_bill"] >= 20) & (tips["total_bill"] <= 30)], x="total_bill", y="tip", hue="day", size="size", palette=["gray"], marker="o", edgecolor="black", linewidth=0.5)


# In[ ]:





# In[ ]:




