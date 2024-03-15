import os
from crewai import Agent, Task, Crew, Process
from crewai import Agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["OPENAI_API_KEY"] = "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"  
os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5'

# Set gemini pro as llm
llm_g = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.5,
                             google_api_key="AIzaSyD-m-O6MT-XUUuG5yte5ZycRHe-qi8oP7c")


#create searches
tool_search = DuckDuckGoSearchRun()


repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm_h = HuggingFaceEndpoint(
    repo_id=repo_id,temperature=0.5
)

# Define Agents
email_author = Agent(
    role='Professional Email Author',
    goal='Craft concise and engaging emails',
    backstory='Experienced in writing impactful marketing emails.',
    verbose=True,
    allow_delegation=False,
    llm=llm_g,
    tools=[]
)
# Initialize ChatGPT 3.5
researcher_llm = ChatOpenAI(model="gpt-3.5-turbo", verbose=True)
# Create the Senior Research Analyst Agent
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory="""You work at a leading tech think tank.
                 Your expertise lies in identifying emerging trends.
                 You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    llm=researcher_llm,
    allow_delegation=False,
    tools=[],
) 
# Initialize ChatGPT 4
writer_llm = ChatOpenAI(model="gpt-3.5-turbo", verbose=True)

# Create the Content Writer Agent
writer = Agent(
    role='Content Writer',
    goal='Craft insightful and engaging articles on AI advancements',
    backstory="""You're an accomplished writer with a passion for technology. Your articles simplify complex concepts for a broad audience.""",
    verbose=True,
    llm=writer_llm,
    allow_delegation=False,
    tools=[],
)  

# Initialize gpt 3.5 turbo
analyst = Agent(
    role='Professional Email Author',
    goal='Craft concise and engaging emails',
    backstory='Experienced in writing impactful marketing emails.',
    verbose=True,
    allow_delegation=False,
    llm=llm_g,
    tools=[]
) 


stylist = Agent(
    role='Data stylist',
    goal='Analyze and interpret complex datasets to forecast AI trends',
    backstory="""With a strong background in statistics and machine learning,you excel at uncovering insights from data to predict future developments.""",
    verbose=True,
    llm=llm_h,
    allow_delegation=False,
    tools=[],
) 
task1 = Task(
    description=""" Identify and summarize the latest research papers on AI and data science that could impact future technology trends.""",
    expected_output="Full analysis report in paragraph form",
    agent=researcher
)

task2 = Task(
    description="""Based on the research analyst's findings, draft an engaging article that highlights key developments in AI and their potential implications.""",
    expected_output="Full analysis report in paragraph form",
    agent=writer
)

task3 = Task(
    description="""Analyze recent data trends in AI technology advancements and provide a forecast report on future directions..""",
    expected_output="Full analysis report in paragraph form",
    agent=analyst
)

task4 = Task(
    description="""Style the blog beautifully with bullets and other features""",
    expected_output="Full analysis report in paragraph form",
    agent=stylist
)

from crewai import Crew, Process

# Define the tasks (assuming task objects have been created)
tasks = [task1,task2,task3,task4]

# Create the crew
ai_crew = Crew(
    agents=[researcher, writer, analyst,stylist],
    tasks=tasks,
    process=Process.sequential,
    verbose=True,
)    

# Kickoff the crew
result = ai_crew.kickoff()
print(result) 