from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel , Field
import os , operator, time


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)



# schema for structured output
class SentimateSchema(BaseModel):
    sentimate : Literal["Positive","Negative"] = Field(description="Feedback on the essay")


class DiagnosisSchema(BaseModel):
     issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in thereview')
     tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
     urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')



structured_model = model.with_structured_output(SentimateSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)

# prompt = """What is the statement of the  following review - The softe wear is the not bad """

# final_sentimate = structured_model.invoke(prompt)

# print (" ... ")
# print(final_sentimate.sentimate)


# define state

class ReviewState(TypedDict):
    review : str
    sentimate : Literal["Positive","Negative"]
    diagnosis : dict
    reponse : str

def find_sentiment(state: ReviewState) -> ReviewState:
    prompt = f"""What is the statement of the  following review - {state['review']}"""
    final_sentimate = structured_model.invoke(prompt)
    sentimate = final_sentimate.sentimate
    return {"sentimate":sentimate}

# conditional function for edge

def check_sentimate(state: ReviewState) ->Literal["positive_response","run_diagnosis"]:

        if state['sentimate'] == "Positive":
            return "positive_response"
        else:
            return "run_diagnosis"
        


def positive_response(state: ReviewState) -> ReviewState:
     prompt = "Thank you for your positive feedback! We're glad you had a good experience."
     response =  model.invoke(prompt)
     return {"response": response}
           
def run_diagnosis(state: ReviewState) -> ReviewState:
     prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
     """
     
     diagnosis = structured_model2.invoke(prompt)
     # model_dump returns a dictionary representation of the pydantic model
     return {"diagnosis": diagnosis.model_dump()}


def negative_response(state:ReviewState)-> ReviewState:
    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message"""
    response = model.invoke(prompt).content
    return {"response": response}   
      
        

graph = StateGraph(ReviewState)

# add nodes to the graph
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)




# define edges
graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentimate)

graph.add_edge("positive_response", END)

graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)


# compile the graph

workflow = graph.compile()

# invoke the graph

intial_state={
    'review': "I am FED UP! I've been waiting for my account verification for 10 days and nobody is responding. Your entire system is a mess and I'm losing money every minute. Fix this RIGHT NOW or I'm deleting my account!"
}


result = workflow.invoke(intial_state)

print (" .\n.. ")
print(result)

