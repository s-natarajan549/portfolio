import pandas as pd 
import numpy as np 
from datasets import load_dataset 

dataset = load_dataset("sjsq/PrivacyPolicy", split="train")
df = dataset.to_pandas()

print(df['Text'].head(20)) 


df.to_csv('privacy_policies.csv')

def main(): 
    return 

if __name__ == "__main__": 
    main() 




