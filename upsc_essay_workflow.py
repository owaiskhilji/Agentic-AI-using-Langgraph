from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel , Field
import os , operator, time


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


class EvaluationSchema(BaseModel):
    feedback : str = Field(description="Feedback on the essay")
    score : int = Field(description="Score out of 10")



structured_model = model.with_structured_output(EvaluationSchema)

essay = """Pakistan has massive potential in AI due to its large youth population and a strong freelancer community. By integrating AI into the IT sector, the country can significantly boost its economy and global tech exports. However, challenges like inconsistent internet and power supply must be addressed to ensure steady progress. Modernizing the education system with a focus on data science is essential for preparing the next generation. If applied to local industries like agriculture, AI can provide innovative solutions to long-standing problems. With the right policies, Pakistan can eventually emerge as a competitive global hub for technology"""

essay2 = """Pakistan big AI power because many young kids and internet workers. IT sector if add AI then money come and export also high. But light and net problem always there, so progress slow. School need data science for kids future. In agriculture also AI good for old problems. Government if make policy then Pakistan technology hub maybe"""



# prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {essay}'

# evaluation = structured_model.invoke(prompt)

# print ("Starting evaluation of the essay... ")
# print(evaluation.feedback)
# print("*" * 50 )
# print(evaluation.score)


# define state  

class UPSCState(TypedDict):
    eassay : str
    language_feedback : str
    analysis_feedback : str
    clarity_feedback : str
    overall_feedback : str
    individual_scores : Annotated[list[int],operator.add]
    avg_score: float






# define functions for nodes

def avluate_language(state: UPSCState) -> UPSCState:

    time.sleep(10)

    prompt = f"Evaluate the language quality of this essay: '{state['eassay']}'. Provide feedback in exactly ONE short sentence and assign a score  out of 10."
    evaluation = structured_model.invoke(prompt)

    return {
        'language_feedback': evaluation.feedback,
        'individual_scores': [evaluation.score]
    }


def evaluate_analysis(state: UPSCState) -> UPSCState:

    time.sleep(10)

    prompt = f"Evaluate the depth of analysis in this essay: '{state['eassay']}'. Provide feedback in exactly ONE short sentence and assign a score out of 10."
    
    evaluation = structured_model.invoke(prompt)

    return {
        'analysis_feedback': evaluation.feedback,
        'individual_scores': [evaluation.score]
    }


def evaluate_thought(state: UPSCState) -> UPSCState:
    time.sleep(10)

    prompt = f"Evaluate the clarity of thought in this essay: '{state['eassay']}'. Provide feedback in exactly ONE short sentence and assign a score out of 10."
    evaluation = structured_model.invoke(prompt)

    return {
        'clarity_feedback': evaluation.feedback,
        'individual_scores': state['individual_scores'] + [evaluation.score]
    }

def final_evaluation(state: UPSCState) -> UPSCState:

    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    overall_feedback = f"The essay received an average score of {avg_score:.2f} out of 10. Language feedback: {state['language_feedback']}. Analysis feedback: {state['analysis_feedback']}. Clarity feedback: {state['clarity_feedback']}."

    return {
        "overall_feedback": overall_feedback,
        "avg_score": avg_score
    }

    


graph = StateGraph(UPSCState)

# add nodes
graph.add_node("evaluate_language",avluate_language)
graph.add_node("evaluate_analysis",evaluate_analysis)
graph.add_node("evaluate_thought",evaluate_thought)
graph.add_node("final_evaluation",final_evaluation)

# add edges parallel evaluation of language, analysis and thought


graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")


graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")


graph.add_edge("final_evaluation", END)


workflow = graph.compile()


initial_state = {
    'eassay': essay2,    
}


result = workflow.invoke(initial_state)


print ("\n\n --- Detailed Feedback --- \n")

print("Language Feedback: ", result['language_feedback'])
print("Analysis Feedback: ", result['analysis_feedback'])
print("Clarity Feedback: ", result['clarity_feedback'])

print("\n\n--- Individual Scores ---\n")
print(result['individual_scores'])
print("\n\n--- Average Score ---\n")
print(result['avg_score'])