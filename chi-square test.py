#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


# In[6]:


xo, xe = [324, 78, 261], [371, 80, 212]
xc = pd.DataFrame([xo, xe], columns=['Item A', 'Item B', 'Item C'], index=['Obs', 'Exp'])
xc


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

ax = xc.plot(kind='bar', title='Number of Items', figsize=(8, 6))
ax.set_ylabel('value')
plt.grid(color='darkgray')
plt.show()


# In[8]:


from scipy.stats import chisquare

result = chisquare(xo, f_exp=xe)
result

# p-value가 유의수준 0.05보다 아주 작은 값이므로 귀무가설을 기각하고 대립가설을 지지합니다.


# In[11]:


plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b--')
plt.axvline(x=x95, color='black', linestyle=':')
plt.text(x95, .4, 'critical value\n' + str(round(x95, 4)), 
         horizontalalignment='center', color='b')

plt.axvline(x=result[0], color='r', linestyle=':')
plt.text(result[0], .4, 'statistic\n' + str(round(result[0], 4)), 
         horizontalalignment='center', color='b')

plt.xlabel('X')
plt.ylabel('P(X)')
plt.grid()
plt.title(r'$\chi^2$ Distribution (df = 2)')
plt.show()


# In[12]:


# 이원 카이제곱검정 (two-way chi-squared test)


# In[14]:


xf, xm = [269, 83, 215], [155, 57, 181]
x = pd.DataFrame([xf, xm], columns=['Item 1', 'Item 2', 'Item 3'], index=['Female', 'Male'])
x


# In[16]:


from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency([xf, xm])

msg = 'Test Statistic: {}\np-value: {}\nDegree of Freedom: {}'
print(msg.format(chi2, p, dof))
print(expected)

# 자유도 (df, degree of freedom)은 (3-1)*(2-1) = 2이고,
# p-value는 유의수준 0.05보다 작은 값이므로 2개 그룹 간에 차이가 있다고 판단할 수 있음

