Llama + chorma db setup

In order to run this project you need to do the following:

### Step 1
Install docker (https://docs.docker.com/get-docker/)

### Step 2
Download the LLama model from Hugging Face (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q2_K.gguf).
Save the llama model in a folder on your computer. I call /path/to/llama/model the absolute path to the folder where the llama model is saved.

### Step 3
Change the docker-compose.yml file to point to the correct location for your llama model, i.e., the folder of the llama-2-7b-chat.Q2_K.gguf file. 
Specifically, replace this line
```    volumes:
      - /Users/andrea/Desktop/PhD/Projects/Current/Reddit/model:/var/model
  ```
with this line:
```    volumes:
      - /path/to/llama/model:/var/model
  ```

### Step 4
Open a terminal and navigate to the folder where the docker-compose.yml file is located. Then run the following command:

```docker-compose up```

Magic should happen and you should see the services starting.

### Step 5
You now have all the services running. 
You can access the llama.cpp api documentation going to http://localhost:5000/docs.

### Step 6
Run the example.py file to see if everything is working. The output should be:
```
{'ids': ['id1'], 'embeddings': [[-0.006452106962426977, -0.007737283456783229, 0.021261541472589872, ... lots of other numbers ..., 0.011253083784333614, -0.015358811976663474]], 'metadatas': [None], 'documents': ['Here is an example sentence'], 'data': None, 'uris': None}
```