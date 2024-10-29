#%%
import pandas as pd
from io import StringIO
import requests

url="https://query.data.world/s/ttvvwduzk3hwuahxgxe54jgfyjaiul"
s=requests.get(url).text
c=pd.read_csv(StringIO(s))
print(c.head())
print(s)
# %%
c.info()
# %%
c.info()
# %%
#drop columns that have more than 10% missing values
c.dropna(thresh=c.shape[0]*0.9,axis=1,inplace=True)

# %%
c.info()
# %%

