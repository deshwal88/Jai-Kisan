import numpy as np
import pandas as pd

df=pd.read_feather('./seeds')
features=['temperature(C)','rainfall(cm)','soil_ph','irrigation','yield(quint/hect)','sowing_time','soil_type','maturity_class']
data=df[features]


def seq_sim(lst,n,a=0.5):
    p,q=lst
    n=int(n)
    if p>q:
        if n>=q or n<=p:
            return 1
        return a**min(n-p,q-n)    
    else:
        if n>=p and n<=q:
            return 1
        return a**min(abs(p-n),abs(n-q))
    
    
def metric(arr1,arr2,cutoff):
    tot=0
    for i in range(0,len(arr1)):
        if i>=cutoff:
            tot+=np.dot(arr1[i],arr2[i])
        else:
            tot+=seq_sim(arr1[i],arr2[i])
    return tot/len(arr1)    

def get_recommendations(inp,n):
    score=data.apply(metric,axis=1,cutoff=6,arr2=inp)
    top=[]
    for i in range(n):
        ind=np.argmax(score)
        score[ind]=0
        top.append(ind)
    return df.iloc[top][['crop_name','variety']].to_dict(orient='records')
        
        
    