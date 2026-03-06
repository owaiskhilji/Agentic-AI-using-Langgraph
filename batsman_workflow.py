from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import os


# define state

class BatsmanState(TypedDict):
    runs : int
    balls : int
    fours : int
    sixes : int

    sr : float
    bpb : float
    boundary_percentage : float
    summary : str




# functions for nodes   

def calculate_sr(state: BatsmanState) -> BatsmanState:
    sr =(state['runs']/state['balls'])*100

    return {
        'sr': sr
    }

def calculate_bpb(state: BatsmanState) -> BatsmanState:
    bpb = state['balls']/(state['fours'] + state['sixes'])

    return {
        'bpb': bpb
    }

def calculate_boundary_percentage(state: BatsmanState) -> BatsmanState:
    total_boundaries = state['fours'] + state['sixes']
    boundary_percentage = (total_boundaries/state['balls'])*100

    return {
        'boundary_percentage': boundary_percentage
    }



def calculate_summary(state: BatsmanState) -> BatsmanState:
    summary = f"Batsman scored {state['runs']} runs off {state['balls']} balls with a strike rate of {state['sr']:.2f}, boundary percentage of {state['boundary_percentage']:.2f}% and a boundary every {state['bpb']:.2f} balls."

    return {
        'summary': summary
    }





graph = StateGraph(BatsmanState)

# add node
graph.add_node("calculate_sr", calculate_sr)
graph.add_node("calculate_bpb", calculate_bpb)
graph.add_node("calculate_boundary_percentage", calculate_boundary_percentage)
graph.add_node("calculate_summary", calculate_summary)


# add edge for parallel execution of sr, bpb and boundary percentage
graph.add_edge(START, "calculate_sr")
graph.add_edge(START, "calculate_bpb")
graph.add_edge(START, "calculate_boundary_percentage")

graph.add_edge("calculate_sr", "calculate_summary")
graph.add_edge("calculate_bpb", "calculate_summary")
graph.add_edge("calculate_boundary_percentage", "calculate_summary")

graph.add_edge("calculate_summary", END)


workflow = graph.compile()


print ("--- Batsman Performance Summary ---")

initial_state = {
    "runs": 100,
    "balls":50,
    "fours": 6,
    "sixes": 4
}

result = workflow.invoke(initial_state)
print(result)

