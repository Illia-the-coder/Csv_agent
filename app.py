import streamlit as st
import pandas as pd
import io
import os
import pandas as pd
from langchain.llms import OpenAI
from langchain.agents import *
from langchain.agents.agent_toolkits import *
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.tools.python.tool import PythonAstREPLTool
from matplotlib import pyplot as plt

def save_to_csv(data):
    df = pd.DataFrame(data, columns=["Question", "Answer"])
    df.to_csv("qa_data.csv", index=False)

def download_csv(data):
    csv = pd.DataFrame(data, columns=["Question", "Answer"]).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="qa_data.csv">Download CSV file</a>'
    return href

class CSVagent:
    def __init__(self, api_key, df):
        os.environ['OPENAI_API_KEY'] = api_key
        self.agent = agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True, max_iterations=2)
    def run(self, query):
        return self.agent.run(query)


def read_file(uploaded_file):
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        file_type = uploaded_file.name.split('.')[-1]
        st.write(file_type)
        try:
            if file_type == 'csv':
                return pd.read_csv(io.StringIO(file_content.decode('utf-8')))
            elif file_type in ['xlsx', 'xls']:
                return pd.read_excel(io.BytesIO(file_content))
            elif file_type == 'json':
                return pd.read_json(io.StringIO(file_content.decode('utf-8')), lines = True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}. Please upload a CSV, Excel, or JSON file.")
        except Exception as e:
            st.info(e)
            return None
        
    else:
        st.info("Please upload a file.")

st.title("File Upload Pandas Viewer")
st.markdown("Upload a CSV, Excel, or JSON file to view its contents using pandas.")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json"])
df = read_file(uploaded_file)
if df is not None:
    st.write(df)
    try:
        # qa_data = []
        AG = CSVagent( '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',df)
        # n=0
        st.warning("Please enter a query.")
        input_value = st.text_input("Enter a value", key=f"input")
        answer = AG.run(input_value)
        st.write("Answer:", answer)

     
    except Exception as InvalidRequestError:
        st.error(InvalidRequestError)
