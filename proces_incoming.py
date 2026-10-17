from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
import pandas as pd
import numpy as np

def create_embeddings_batch(text_list):
    batch_size = 20
    all_embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        sub_list = text_list[i:i + batch_size]
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": sub_list
        })
        # API response se embeddings extract karke add karein
        res = r.json()
        all_embeddings.extend(res.get("embeddings", []))
        
    return all_embeddings  # Return karna mandatory hai

def inference(prompt,model):
    r = requests.post("http://localhost:11434/api/generate", json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream":  False
            })
    response = r.json()
    print(response)
    return response
    


df = joblib.load('embeddings.joblib')
incoming_query = input("Ask a question: ")
question_embedding = create_embeddings_batch([incoming_query])[0]

# Vector shapes format for Cosine Similarity
doc_embeddings = np.vstack(df['embedding'].values)
q_embedding = np.array(question_embedding).reshape(1, -1)

similarities = cosine_similarity(doc_embeddings, q_embedding).flatten()

top_results = 5
max_indx = similarities.argsort()[::-1][:top_results]

new_df = df.iloc[max_indx]
#print(new_df[["title", "text", "number"]])

prompt = f''' i am teaching java in my java course devlepment. Here are video subtitle chunks contatting video title,
video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient = "records")}
----------------------
{incoming_query}
usre ask this question related to the chunks,you have to answer in a human way (dont mention the above format,its just for you)
where and how much content in tougth in which 
viddeo(in which video and at what timestamp) and guide the user to go to the particular video. if ask unrelated question 
related to the course'''

with open("prompt.text","w")as f:
    f.write(prompt)
    
    response = inference(prompt,model="llama3.2")["response"]
    print(response)
    with open("response.text","w")as f:
        f.write(response)
    

#for index, item in new_df.iterrows():
    #print(index,item["title"], item["number"], item["text"],item["start"],item["end"])
