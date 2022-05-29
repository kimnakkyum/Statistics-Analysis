#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import f


# # 독립표본 t-test

# In[4]:


a = [-1,0,3,4,1,3,3,1,1,3]
b = [6,6,8,8,11,11,10,8,8,9]
group = ['a']*10 + ['b']*10
data = pd.DataFrame({'group':group, 'temp':a+b})
data.head(3)


# In[9]:


plt.figure(figsize=(6,6))
sns.boxplot(x='group', y='temp', data=data)
plt.title('Box plot')
plt.show()


# In[13]:


# 데이터가 10개뿐이므로 shapiro-wilks의 정규성을 검정해보자
normal1 = shapiro(a)
normal2 = shapiro(b)
print(normal1, normal2)

#결과는 모두 p-value가 0.05보다 커서 정규성을 만족한다.


# In[14]:


#levene test로 등분산성을 검정
from scipy.stats import levene, ttest_ind
print(levene(a,b))

#등분산성을 bartlett test로 할 수도 있음 바틀렛
from scipy.stats import bartlett
print(bartlett(a,b))

#p-value가 유의수준 0.05보다 크기 때문에 귀무가설을 기각하지 않는다
#따라서 a,b두 집단의 데이터는 등분산성을 만족한다고 볼 수 있음. 


# In[15]:


ttest_ind(a,b)

#검정 통계량은 -8.806, p-value는 6.085e-08이다. p-value가 0에 가까운 매우 작은 숫자로 
#유의수준보다 작기 때문에 귀무가설을 기각한다. 
#따라서 a,b두 지역의 겨울 낮 최고기온에는 통계적으로 유의한 차이가 존재한다는 결론을 내릴 수 있음


# In[20]:


#등분산성을 만족하지 못하는 2개의 그룹에 대한 ttest_ind()에는 equal_var=False 옵션을 추가합니다.

ttest_ind(a,b, equal_var=False)


# # 대응표본 t-test

# In[22]:


x1 = [.430,.266,.567,.531,.707,.716,.651,.589,.469,.723]
x2 = [.415,.238,.390,.410,.605,.609,.632,.523,.411,.612]


# In[23]:


# 박스플롯 작성하기

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(6, 6))
plt.grid()
plt.boxplot([x1, x2])
plt.xlabel('when')
plt.ylabel('score')
plt.title('Box Plot')
plt.grid()
plt.show()


# In[27]:


# 데이터 정규성 검정하기

from scipy.stats import shapiro

normal1 = shapiro(x1)
normal2 = shapiro(x2)
print(normal1, normal2)

# p-value가 모두 0.05보다 크기 때문에 정규성 문제가 없다.


# In[31]:


# F test to compare two variances 수행하기

df1, df2 = len(x1) - 1,len(x2) - 1
v = (np.var(x1, ddof=1), np.var(x2, ddof=1))
F = max(v) / min(v)
cdf = f(df1, df2).cdf(F)
p_value = 2 * min(cdf, 1 - cdf)
print(F, p_value)

# p-value가 0.7441로 등분산성이 있다고 판단된다.


# In[34]:


# t-test 수행하기

from scipy.stats import ttest_rel

ttest_rel(x1, x2)

# p-value가 0.05보다 작아서 귀무가설을 기각할 수 있다.
# 즉, 2개의 그룹의 평균 차이는 통계적으로 유의하다.

