# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 14:27:15 2014

@author: mohith
"""

import json
import random
import re
data=json.loads(open('all_data.json').read())
X=[]
Y=['Org','Family','Other']
def modifytag(tag):
    tag=str(tag)
    if tag=='Model':
        return 'Family'
    elif tag!='Family' and tag!='Org':
        return 'Other'
    return tag
        
for i in data['root']:
    for j in i['data']:
        temp=[]
        for k in j['updates']:
                temp.append(str(k['word']))
        for k in range(len(j['updates'])):
            if k==0:
                X.append((None,None,temp,k,modifytag(j['updates'][k]['tag'])))
            elif k==1:
                X.append((None,modifytag(j['updates'][k-1]['tag']),temp,k,modifytag(j['updates'][k]['tag'])))
            else:
                X.append((modifytag(j['updates'][k-2]['tag']),modifytag(j['updates'][k-1]['tag']),temp,k,modifytag(j['updates'][k]['tag'])))

def features():                
	def f1(X,tag):
		if tag=='Family' and X[1]=='Family':
			return 1
		return 0
    
	def f2(X,tag):
		if tag=='Family' and X[0]=='Org':
			return 1
		return 0
    
	def f3(X,tag):
		if tag=='Family' and (X[3] != len(X[2])-1) and X[2][X[3]+1]=='is':
			return 0
		return 1
    
	def f4(X,tag):
		if tag=='Family' and (len(X[2][X[3]])>1) and X[2][X[3]][1].isupper():
			return 1
		return 0
    
	def f5(X,tag):
		if tag=='Org' and (X[1]=='Other' or X[1]==None):
			return 1
		return 0
        
	def f6(X,tag):
		if tag=='Org' and X[1]=='Org':
			return 0
		return 1
    
	def f7(X,tag):
		if tag=='Org' and X[2][X[3]][0].isupper():
			return 1
		return 0
        
	def f8(X,tag):
		if tag=='Other' and X[3]!=0 and X[2][X[3]].islower():
			return 1
		return 0
    
	return [f1,f2,f3,f4,f5,f6,f7,f8]
    

class MyMaxEnt():
    def __init__(self,X,features,tags,train_size):
        self.X=X
        self.features=features
        self.model=[0]*len(features)
        self.tags=tags
        self.train_size=train_size
        
    def create_dataset(self):
        random.shuffle(self.X)
        self.trainset=self.X[0:self.train_size]
        self.testset=self.X[self.train_size:2*self.train_size]
        
    def cost(self,model):
        import math
        value1,value2=0,0
        for i in self.trainset:
            for j in range(len(self.features)):
                value1+=model[j]*self.features[j](i,i[4])
        for i in self.trainset:
            for j in range(len(self.features)):
                temp=0
                for k in self.tags:
                    temp+=math.exp(model[j]*self.features[j](i,k))
                value2+=math.log1p(temp)
        return value2-value1
    
    def train(self):
        from scipy.optimize import minimize
        params=minimize(self.cost,self.model,method = 'L-BFGS-B')
        print params
        self.model=params.x
        
    def p_y_given_x(self,h,tag):
        import math
        numerator,denominator=0,0
        for i in range(len(self.features)):
            numerator+=self.model[i]*self.features[i](h,tag)
        for i in self.tags:
            temp=0
            for j in range(len(self.features)):
                temp+=self.model[j]*self.features[j](h,i)
            denominator+=math.exp(temp)
        return math.exp(numerator)/denominator
        
    def classify(self,h):
        classified=max([(self.p_y_given_x(h,y),y) for y in self.tags])[1]
        return classified
        
    def classifytest(self):
        result=[(self.classify(x),x[4]) for x in self.testset]
        results={}
        results['Precision']=sum([float(result.count((y,y))+1)/(sum([result.count((y,x)) for x in self.tags])+1) for y in self.tags])/3
        results['Recall']=sum([float(result.count((y,y))+1)/(sum([result.count((x,y)) for x in self.tags])+1) for y in self.tags])/3
        results['F1']=(2*results['Precision']*results['Recall'])/(results['Precision']+results['Recall'])
        return results
        

f=features()       
mymaxent=MyMaxEnt(X,f,Y,50)
mymaxent.create_dataset()
mymaxent.train()
results = mymaxent.classifytest()

print "precision=",results['Precision']
print "recall=",results['Recall']
print "F1=",results['F1']
		
