# Task 1
---
## Level 1: Variable Identification Protocol 

#### Imported necessary libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
#### Using EDA techniques
* Plotting Histograms
``` python
features = [data['Feature_1'], data['Feature_2'] , data['Feature_3']]

plt.Figure(figsize=(5,5))
plt.hist(features[0] , bins=20 , color='blue')
plt.ylabel('frequency')
plt.xlabel('features_1')
plt.title('histogram of feature 1')
```

--- 

#### Plotting Correlation Heatmap of numeric data
```python
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, linecolor='white', square=True)
plt.title('Correlation Heatmap')
```


#### Boxplot
```python
sns.boxplot(x='romantic', y=features[0], data=data)
plt.title('feature_1 vs Romantic Relationship Status')
plt.xlabel('Romantic Relationship (yes/no)')
plt.ylabel('feature_1')
```


#### Conclusion :
* Feature_1 may represent age .

* Feature_2 may represent screen time hours because positive correlation with absences .

* Feature_3 may represent romantic satisfication score beacuse students who are in a romantic relationship score significantly higher in this feature.

---



##  Level 2: Data Integrity Audit

#### Calculating number of missing values
```python
data.isnull().sum()
```
#### Filling missing values.

1. 'famsize' represents no. of family members having categorical type of data
so filling it with Most common value (mode).
```python
data['famsize'].fillna(data['famsize'].mode()[0], inplace = True)
```
2. 'higher' have discrete data values 'Yes' or 'No' so filling with mode will be best.
```python
data['higher'].fillna(data['higher'].mode()[0], inplace = True)
```
3. 'Feature_1' and 'G2' have continuous type of data so using mean.
```python
for col in ['Feature_1','G2']:
    data[col].fillna(data[col].mean(), inplace=True)
```
4. Fedu, traveltime, absences, freetime, Feature_2, Feature_3 all have discrete type data.Numeric but have maximum and minimum values so filling with median will make it more stable.
```python
for col in ['Fedu', 'traveltime', 'absences', 'freetime', 'Feature_2', 'Feature_3']:
    data[col].fillna(data[col].median(), inplace=True)
```

##  Level 3: Exploratory Insight Report 

#### Questions :
1. Does drinking alcohol affect how well students do in academics ?
```python
plt.figure(figsize=(5, 5))
sns.violinplot( x='Dalc', y='G3', data= data, palette='Set2')
plt.title('Grades vs alcohol consumption')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Grade (G3)')
```


---


2. Are students from urban areas more likely to be in romantic relationships than those from rural areas?
```python
in_relationship = data[data['romantic']=='Yes']
counts = data['address'].value_counts()

plt.figure(figsize=(5,5))
plt.bar(counts.index, counts.values, color=['blue', 'red'])
plt.title('Number of Students in Relationship by Address')
plt.xlabel('Address')
plt.ylabel('Number of Students')
```


---

3. How does parental cohabitation status relate to stress levels among students?

```python
together_parents = data.loc[data['Pstatus'] == 'T', 'Feature_2'].value_counts().sort_index()
separated_parents = data.loc[data['Pstatus'] == 'A', 'Feature_2'].value_counts().sort_index()

plt.subplot(1, 2, 1)
plt.bar(together_parents.index , together_parents.values ,color='blue')
plt.title('Stress Levels for  Together')
plt.xlabel('Stress Level')
plt.ylabel('Number of Students')
plt.xticks(features[1])

plt.subplot(1, 2, 2)
plt.bar(separated_parents.index , separated_parents.values , color='red')
plt.title('Stress Levels for Separated')
plt.xlabel('Stress Level')
plt.ylabel('Number of Students')
plt.xticks(features[1])

plt.tight_layout()
```


---

4. Does internet access at home affect academic performance?
```python
sns.violinplot(x='internet', y='G3', data=data)
plt.title("Grades by Internet Access at Home")
plt.xlabel('Internet Access')
plt.ylabel('Grades')
plt.title('Grades by Internet Access at Home')
```



---

5.  Does being in a romantic relationship impacts on student's final grades?
```python
plt.figure(figsize=(5, 5))
sns.boxplot( x='romantic', y='G3',data=data , palette='pastel')
plt.title("Romantic Relationship vs Final Grades")
plt.xlabel("Relationship status")
plt.ylabel("Final Grade")
```


---

##  Level 4: Relationship Prediction Model    

Importing Libraries
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```
Encoding the data
```python
X = data.iloc[: , : -1]
Y = data.loc[: , 'romantic']

X = pd.get_dummies( X)
```
splitting the data
```python
X_train , X_test , Y_train , Y_test =train_test_split(X ,Y , test_size=0.4 ,random_state=99)
```
Implementing the models
```python
LogisticRegression_model = LogisticRegression()
Random_forest_model = RandomForestClassifier()

LogisticRegression_model.fit(X_train ,Y_train)
Random_forest_model.fit(X_train ,Y_train)

y_pred_logreg = LogisticRegression_model.predict(X_test)
y_pred_rf = Random_forest_model.predict(X_test)
```
Calculating accuracy of models
```python
print( "accuracy of Logistic regression model : ",accuracy_score(Y_test ,y_pred_logreg) * 100)
print( "accuracy of Random forest model : ",accuracy_score(Y_test ,y_pred_rf) * 100)
```
Weightage of features
```python
imp_features_Logreg_model = pd.Series(abs(LogisticRegression_model.coef_[0]), index=X.columns)
imp_features_Logreg_model_order= imp_features_Logreg_model.sort_values(ascending=False)

imp_features_rf_model = pd.Series(Random_forest_model.feature_importances_, index=X.columns)
imp_features_rf_model_order= imp_features_rf_model.sort_values(ascending=False)

print('IMP features of Logistic_regression_model\n' ,imp_features_Logreg_model_order,'\n\n')
print('IMP features of Random forest_model\n',imp_features_rf_model_order)
```

---

##  Level 5: Model Reasoning & Interpretation     

#### Decision boundary for 'Feature_1' and 'goout'
```python
data['romantic'] = data['romantic'].map({'yes': 1, 'no': 0})

X = data[['Feature_1', 'goout']].values
y = data['romantic'].values

# creating meshgrid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid = np.c_[xx.ravel(), yy.ravel()]
```
###### For RandomForest_model
```python
rf_model = RandomForestClassifier()
rf_model.fit(X, y)
Z =rf_model.predict(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(5, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.xlabel("Feature_1")
plt.ylabel("goout")
plt.title("Decision Boundary for Feature_1 and goout")
```

##### For Logistic regression model
```python
lrg_model = LogisticRegression()
lrg_model.fit(X, y)
p =lrg_model.predict(grid)
p = p.reshape(xx.shape)

plt.figure(figsize=(5, 5))
plt.contourf(xx, yy, p, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.xlabel("Feature_1")
plt.ylabel("goout")
plt.title("Decision Boundary for Feature_1 and goout")
```


---

### Using Shap for logistic regression model

```python
import shap

lrg_explainer = shap.Explainer(LogisticRegression_model, X_train)
lrg_shap_values = lrg_explainer(X_train)

shap.plots.bar(lrg_shap_values)
```

```python
idx_yes = np.where(y_pred_logreg=="yes" )[0][0]
idx_no = np.where(y_pred_logreg == "no")[0][0]


print("SHAP Explanation for a student predicted YES:")
shap.plots.waterfall(lrg_shap_values[idx_yes])

print("SHAP Explanation for a student predicted NO:")
shap.plots.waterfall(lrg_shap_values[idx_no])
```


---
---


# Task 2 
---
### Installations 
```python
!pip install -q langgraph
!pip install -q langchain langchain-google-genai google-generativeai
!pip install pytrends
!pip install -U langchain langchain-community
!pip install wikipedia
```
### Gemini chatbot setup
```python

from langchain_google_genai import ChatGoogleGenerativeAI

Gemini_chatbot = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key="AIzaSyAfAqf22-jWlSyx5lDCDkM6_CmW9pP3Sq8")
```
### Defining State
```python
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    messages : Annotated [list[AnyMessage],operator.add]
```
### Chatbot Node
```python
def chatbot_Node(state:State) -> State:
    messages = state['messages']
    output = Gemini_chatbot.invoke(messages)
    return {"messages": messages + [output]}
```
### Calculator Node
```python
def calculator_node(state : State) -> State:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message , HumanMessage):
          expression = last_message.content
          try:
            expression = last_message.content
            answer = eval(expression)
            reply = AIMessage(content=str(answer))
          except:
            reply = AIMessage(content="sorry, I can't calculate that")     
   
          return {"messages": messages +[reply]}
    return {"messages": messages}
```
### Langgraph Formation
```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)

graph.add_node("Gemini_chatbot", chatbot_Node)
graph.add_node("calculator", calculator_node)

graph.add_edge(START, "Gemini_chatbot")
graph.add_edge("Gemini_chatbot", "calculator")
graph.add_edge("calculator", END)

chain = graph.compile()
```
### Visualization
```python
from IPython.display import display, Image
import matplotlib.pyplot as plt
import networkx as nx

display(Image(chain.get_graph().draw_mermaid_png()))
```

## Level 2: Senses of the World   

### Fashion Recommender Tool:
```python
from langchain.tools import tool
from pytrends.request import TrendReq
import spacy

nlp = spacy.load("en_core_web_sm")

@tool
def Fashion_recommender(query: str) -> str:
    """Given a user's query, extract a location and return top fashion trends based on Google Trends."""

    doc = nlp(query)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    if not locations:
        return "Sorry, I can't detect location."

    city = locations[0]


    pytrends = TrendReq(hl='en-US', tz=360)
    keywords = [f"fashion {city}", f"shoes {city}", f"clothing {city}", f"style {city}"]

    pytrends.build_payload(keywords)
    data = pytrends.interest_over_time()

    if data.empty:
        return f"No recent fashion trends found."

    latest = data.iloc[-1][keywords].sort_values(ascending=False)
    top_trends = latest.head(3).index.tolist()

    cleaned_trends = [kw.replace(f" {city}", "") for kw in top_trends]

    return f"Trending fashion in {city} :{', '.join(cleaned_trends)}"
```
### Weather Extractor Tool
```python
import requests

def weather_extractor_tool(location: str) -> str:
    api_key = "7b6ee732596e9a0d8259a9fd7cd1c5b7" 
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200 or "weather" not in data:
        return f"not get weather for {location}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"The weather in {location} is {desc} with a temperature of {temp}°C."
```
### Binding Tools
```python
tools.append(Fashion_recommender)
tools.append(weather_extractor_tool)
tool_names = {t.name: t for t in tools}

Gemini_chatbot_tool = Gemini_chatbot.bind_tools(tools)
```
##  Level 3: Judgement and Memory  
### Tool function
```python
def tool_agent_node(state: State) -> State:
    messages = state["messages"]
    response = Gemini_chatbot_tool.invoke(messages)
    return {"messages": messages + [response]}
```
### Router logic
```python
def router_node(state: State) -> str:
    last_message = state["messages"][-1].content.lower()

    if any(keyword in last_message for keyword in ["weather", "temperature", "forecast", "rain", "sunny", "humidity"]):
        return "tool_agent"
    elif any(keyword in last_message for keyword in ["fashion", "style", "wear", "clothes"]):
        return "tool_agent"
    elif any(symbol in last_message for symbol in ["+", "-", "*", "/", "calculate"]):
        return "tool_agent"
    else:
        return "chatbot"
```
### New Langgraph formation
```python
graph_builder = StateGraph(State)
graph_builder.add_node("router", router_node)
graph_builder.add_node("tool_agent", tool_agent_node)
graph_builder.add_node("chatbot", chatbot_Node)

graph_builder.add_edge(START , "router")
graph_builder.add_edge("router", "tool_agent")
graph_builder.add_edge("router", "chatbot")
graph_builder.add_edge("tool_agent", END)
graph_builder.add_edge("chatbot", END)


#new_chain = graph_builder.compile()
```
#  Level 4: The Architect’s Trial – Multi-Agent Evolution
### Research paper tool
```python
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from pydantic import BaseModel, Field

def get_arxiv_data(query):
    data = arxiv_tool.invoke(query)
    return data

class ArticleTopic(BaseModel):
  topic: str = Field(description="The topic of the article to search on arxiv.")

@tool (args_schema=ArticleTopic)
def arxiv_search(topic: str) -> str:
  """Returns the information about research papers from arxiv"""
  return get_arxiv_data(topic)
```
### Wikipedia tool
```python
import wikipedia

def get_wiki_data(topic):
    data = wikipedia.summary(topic)
    return data

class WikipediaTopic(BaseModel):
  topic: str = Field(description="The wikipedia article topic to search")

@tool(args_schema = WikipediaTopic)
def wikipedia_search(topic: str) -> str:
  """Returns the summary of wikipedia page of the passed topic"""
  return get_wiki_data(topic)
```
### Binding tools of each agent
```python
tools.append(arxiv_search)
tools.append(wikipedia_search)
Gemini_research = Gemini_chatbot.bind_tools([arxiv_search, wikipedia_search])
Gemini_weather = Gemini_chatbot.bind_tools([weather_extractor_tool])
Gemini_fashion = Gemini_chatbot.bind_tools([Fashion_recommender])
Gemini_calc = Gemini_chatbot.bind_tools([calculator_tool])
```
### Defining agents
```python
def research_agent(state: State) -> State:
    response = Gemini_research.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def weather_agent(state: State) -> State:
    response = Gemini_weather.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def fashion_agent(state: State) -> State:
    response = Gemini_fashion.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def calculator_agent(state: State) -> State:
    response = Gemini_calc.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def fallback_agent(state: State) -> State:
    fallback_msg = AIMessage(content="I'm not sure how to help with that. Can you ask something else?")
    return {"messages": state["messages"] + [fallback_msg]}
```
### Routing logic
```python
def router_2(state: State) -> str:
    last_input = state['messages'][-1].content.lower()
    if any(x in last_input for x in ["exit", "quit", "bye"]):
        return END
    elif any(x in last_input for x in ["arxiv", "research", "paper"]):
        return "research_agent"
    elif any(x in last_input for x in ["weather", "temperature"]):
        return "weather_agent"
    elif any(x in last_input for x in ["fashion", "style", "clothes"]):
        return "fashion_agent"
    elif any(x in last_input for x in ["calculate", "+", "-", "*", "/"]):
        return "calculator_agent"
    else:
        return "fallback_agent"


agents_list = ["research_agent","weather_agent","fashion_agent","calculator_agent","fallback_agent"]
```
### Langgraph formation
```python
builder = StateGraph(State)

builder.add_node("router_2", router_2)
builder.add_node("research_agent", research_agent)
builder.add_node("weather_agent", weather_agent)
builder.add_node("fashion_agent", fashion_agent)
builder.add_node("calculator_agent", calculator_agent)
builder.add_node("fallback_agent", fallback_agent)


builder.add_edge(START, "router_2")
builder.add_conditional_edges("router_2",router_2 , {agent :agent for agent in agents_list})

for agent in agents_list:
    graph_builder.add_edge(agent, "router_2")

builder.set_entry_point("router_2")

Multi_agent_chain = builder.compile()
```


