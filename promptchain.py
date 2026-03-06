from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# define state
class BlogState(TypedDict):
    title : str
    outline : str
    content : str

def create_outline(state: BlogState) -> BlogState:

    title = state['title']

    prompt = f"Create a very brief outline (max 3 bullet points) for a blog on: {title}"
    outline = model.invoke(prompt).content

    state['outline'] = outline

    return state



def create_blog(state: BlogState) -> BlogState:

    title =state['title']
    outline = state['outline']
    prompt = f"""
    Write a short blog post (max 150 words) for the title: {title}.
    Use this outline:
    {outline}
    Keep it concise and professional.
    """
    content = model.invoke(prompt).content

    state['content'] = content
    return state

# create graph
graph = StateGraph(BlogState)

# add node
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)

# add edge
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog") 
graph.add_edge("create_blog", END)

# compile graph to workflow

workflow = graph.compile()

intial_state = {'title': 'Rise of AI in Pakistan'}
result = workflow.invoke(intial_state)
print(result['outline'])


print("--- OUTLINE ---")
print(result['outline'])
print("\n--- BLOG CONTENT ---")
print(result['content'])