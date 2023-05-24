import numpy as np
import pandas as pd
import openai
import os
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer, util


# Load knowledge base data
df_knowledge = pd.read_csv("source_data/foundeverafter.csv")
df_knowledge = df_knowledge.fillna('none')
df_knowledge.dropna(inplace=True)
df_knowledge.reset_index(level=0, inplace=True)

# Text preparation
emb_model=SentenceTransformer(
    #"sentence-transformers/multi-qa-distilbert-cos-v1"
    "all-mpnet-base-v2"
    )

# Function to extract BERT embeddings for text as a list
def calc_embeddings(some_text):
    text_embeddings = emb_model.encode(some_text,normalize_embeddings=True)
    return text_embeddings.tolist()
# calc_embeddings('Sitel Group is changing from using the Duo App on your smart phone')

# Function to create embeddings for each item in a list (row of a df column)
def embedding_list(df_column):
    column_embeddings_list = list(map(calc_embeddings, df_column))

    return column_embeddings_list

#Create embeddings for each column we want to compare our text with
embeddings_Description = embedding_list(df_knowledge['Description'])
embeddings_Link        = embedding_list(df_knowledge['Link'])

# Option to save embeddings if no change rather than re calc everytime
np.save('embeddings/embeddings_Description.npy', np.array(embeddings_Description))
np.save('embeddings/embeddings_Link.npy', np.array(embeddings_Link))

# Option to load saved embeddings if no change rather than re calc everytime
embeddings_Link        = np.load('embeddings/embeddings_Link.npy', allow_pickle= True).tolist()
embeddings_Description = np.load('embeddings/embeddings_Description.npy', allow_pickle= True).tolist()

# Calculate CosSim between question embeddings and article embeddings
def cos_sim_list(embedding_question,embedding_list):
    list_cos_sim = []
    for i in embedding_list:
        sim_pair = util.cos_sim(embedding_question,i).numpy()
        list_cos_sim.append(sim_pair[0][0])
        
    return list_cos_sim

#Calculate outliers within cos_sim_max data set, identified as possible answers
def find_outliers_IQR(cos_sim_max):
   q1=cos_sim_max.quantile(0.25)
   q3=cos_sim_max.quantile(0.75)
   IQR=q3-q1
   outliers = cos_sim_max[((cos_sim_max>(q3+1.5*IQR)))]

   return outliers

#max token limit 4096

#calculate: question embeddings, cosSim with articles, identify 'outliers', create DF of potential answers

def K_BOT(input_question):
    pd.set_option('display.max_colwidth', 5000)

    #question embeddings
    embeddings_q = calc_embeddings(input_question)

    #calculate cosSim for included fields
    cos_sim_max = list(map(max, cos_sim_list(embeddings_q,embeddings_Link),
                                cos_sim_list(embeddings_q,embeddings_Description)))
    df_knowledge['cos_sim_max'] = cos_sim_max

    #calculate log cosSim
    cos_sim_log = np.log2(df_knowledge['cos_sim_max']+1)
    df_knowledge['cos_sim_log'] = cos_sim_log

    #Identify outliers
    df_outliers = find_outliers_IQR(df_knowledge['cos_sim_log']).to_frame().reset_index(level=0, inplace=False)
    
    #Create df of potential answers
    df_answers = df_knowledge[['index','Link','Description','cos_sim_max','cos_sim_log',]].sort_values(by=['cos_sim_max'], 
                                                                        ascending = False).head(len(df_outliers['index']))
    
    #search_results = []
    return df_answers

#Initialise and reset variables, run this once before starting a new chat session
question_summary = ''
history = []
conversation_summary = ''
transcript = ''
knowledge = ''

class ActionChatGPT(Action):
    def name(self) -> Text:
        return "action_chatgpt"
    

    ## ChatGPT

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        input_txt = tracker.latest_message.get('text')
        openai.api_key = os.environ["OPENAI_API_KEY"]



        # Function to summarise the user sequence into a concise string of key words for searching the KB
        def summarise_question(prompt):
            messages = [{"role": "system", "content" :"convert the concepts below into a concise string of key words which would work well as search criteria\n\n"},
              {"role": "user", "content" : prompt}
                ]
  
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            temperature = 0,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            #stop=["."],
            messages = messages
                        )
        
            return completion.choices[0].message.content
        
        # Create a summary of the converstion so far to retain context of the conversation (understand back references from the user)
        def summarise_history_3_5(transcript):
            messages = [{"role": "system", "content" :"summarise the following conversation between the user asking questions and the answers you provided\n\n"
                        +"Context\n\n" + transcript},
                        ]
            
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                temperature = 0,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                #stop=["."],
                messages = messages
                            )
                    
            return completion.choices[0].message.content
        

        # This is the function which produces a response to the users question
        def run_prompt_3_5(prompt,context,summary):

            messages = [{"role": "system", "content" :"you are EverConnect, find an answer in the following knowledge base, feel free to ask questions and make suggestions, reference your answers where possible with the document id\n\n" + context + '\n\n' + summary + '\n\n'},
                
                        {"role": "user", "content" :"please answer my question"},
                        {"role": "assistant", "content" :"here are some suggestions, you can find more info here [document ID]"},

                        {"role": "user", "content" : prompt}
                        ]
            
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                temperature = 0,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                #stop=["."],
                messages = messages
                            )
                    
            return completion.choices[0].message.content
        
        
        #Take the users side of the conversation and summarise into a coherent question (as the chat evolves)
        #input_txt = input('Ask your question...')
        global question_summary
        question_summary = question_summary + '. ' + input_txt

        search_txt = summarise_question(question_summary)
        print(search_txt )

        #Search and return relevant docs from the knowledge base
        df_answers = K_BOT (search_txt)

        #Convert relevant knowledge items into a 'table' to be included as context for the prompt
        knowledge = ''
        for index, row in df_answers.iterrows():
            knowledge = knowledge + 'knowledge ID - ' + str(row['index']) + ' \ttitle - '  + row['Link'] + ' \tdescription - '  + row['Description'] + '\n'
        

        #Come up with a response to the question
        # data = run_prompt_3_5(input_txt, knowledge, conversation_summary).split('\n')
        # while("" in data):
        #     data.remove("")
        global conversation_summary
        data = run_prompt_3_5(input_txt, knowledge, conversation_summary).split('\n')
        while("" in data):
             data.remove("")
        data = (" ").join(data)

        #add Q&A to a list tracking the conversation
        history.append({"role": "user", "content" :input_txt}) 
        history.append({"role": "assistant", "content" :data[0]})
        print(history)

        #Format the list as text to feed back to GPT summary function
        x=0
        transcript =''
        for i in history:
            text = history[x]['role'] + ' - ' + history[x]['content']
            transcript = transcript + text +'\n'
            x=x+1
        transcript

        # response = openai.ChatCompletion.create(
        #     model = "gpt-3.5-turbo",
        #     messages=[
        #     {"role": "system", "content": "You are a helpful and professional assistant"},
        #     {"role": "user", "content": prompt},
        # ]
        # )
        # chat_response = response['choices'][0]['message']['content']
        dispatcher.utter_message(data)
        #summarise transcription for question answer function (this is after the results to reduce wait time)
        conversation_summary = summarise_history_3_5(transcript)
        print(conversation_summary)

        return []
    

   
 