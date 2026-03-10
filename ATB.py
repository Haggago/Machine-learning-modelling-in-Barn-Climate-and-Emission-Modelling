#2019 At Leibniz Institute of Agricultural Engineering and Bio-economy e.V. (ATB)
# coding: utf-8

# In[ ]:


#The estimated regression function of pressure drop for porous media equation (∆𝑃/𝑙=(𝜇𝐷+1/2 𝐹𝜌|𝑢|)𝑢) is a polynomial of degree 2: 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥 + 𝑏₂𝑥² 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

x = np.array([0,0.1,0.2,0.5,0.8, 1.2, 1.5]).reshape(-1, 1)
y = np.array([0, 0.00073456, 0.00282897, 0.0176148, 0.0448887, 0.0985493, 0.151895]).reshape(-1, 1)


# In[ ]:


transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
print(x_)


# In[ ]:


model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)
print('score of linear regression:', r_sq) #R2 score of linear regression


# In[ ]:


print('𝑏₀:', model.intercept_) #𝑏₀ 


# In[ ]:


print('𝑏₁,𝑏₂:', model.coef_) #𝑏₁,𝑏₂ or (a,b as in your pressure drope equation)


# In[ ]:


rmse = np.sqrt(mean_squared_error(x, y))
print('RMSE of polynomial regression:', rmse) #Root Mean Square Error (RMSE) of linear regression


# In[ ]:


x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
print(x_)


# In[ ]:


y_pred = model.intercept_ + model.coef_ * x  #If you want to get the predicted response
y_pred


# In[ ]:


plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
#plt.title('plotted x_ and y_pred values')
plt.legend(['Regression line'])
plt.savefig('linear_regression_fitting.png', dpi=300, bbox_inches='tight')
plt.show()
#Data set in blue, Regression line in red

