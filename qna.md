# QnA app


## Overview

The `app.py` file implements a clinical assistant chatbot using a multi-agent architecture based on Large Language Models (LLMs). This chatbot is designed to handle various types of queries, filter inappropriate or irrelevant questions, and provide accurate and detailed responses in a clinical setting.

## Architecture

The chatbot employs a multi-agent architecture consisting of the following agents:
1. **Filtering_Agent**: Filters out inappropriate, irrelevant, or malicious queries.
2. **Query_Agent**: Queries a vector store to find relevant documents and analyzes their relevance.
3. **Answering_Agent**: Generates responses based on the provided context, document analysis, and conversation history.
4. **Head_Agent**: Coordinates the other agents and manages the overall workflow of the chatbot.

## Classes

### AgentStyle
Defines the conversation style of the chatbot. The available styles are:
- `CHATTY`: The chatbot provides verbose and engaging responses.
- `PRECISE`: The chatbot provides detailed and accurate responses.

### Filtering_Agent
Filters out queries that are inappropriate, irrelevant, or contain prompt injection attempts.

#### Methods
- `__init__(self, client)`: Initializes the agent with an OpenAI client.
- `set_prompt(self, prompt)`: Sets the prompt for the agent.
- `get_action(self)`: Returns the detected action.
- `extract_action(self, response)`: Extracts the action from the response.
- `check_query(self, query)`: Checks the query for any issues and sets the appropriate action.

### Query_Agent
Queries a vector store to find relevant documents and analyzes their relevance to the query.

#### Methods
- `__init__(self, pinecone_index, openai_client, embeddings)`: Initializes the agent with a Pinecone index, OpenAI client, and embeddings.
- `query_vector_store(self, query, k=5)`: Queries the vector store for relevant documents.
- `set_prompt(self, prompt)`: Sets the prompt for the agent.
- `extract_action(self, documents, query=None)`: Analyzes the relevance of the documents to the query.

### Answering_Agent
Generates responses based on the provided context, document analysis, and conversation history.

#### Methods
- `__init__(self, openai_client, mode=AgentStyle.PRECISE)`: Initializes the agent with an OpenAI client and conversation mode.
- `generate_response(self, query, docs, doc_analysis, conv_history, k=5)`: Generates a response based on the query, documents, document analysis, and conversation history.

### Head_Agent
Coordinates the other agents and manages the overall workflow of the chatbot.

#### Methods
- `__init__(self, openai_key, pinecone_key, pinecone_index_name, mode=AgentStyle.PRECISE)`: Initializes the agent with API keys, Pinecone index name, and conversation mode.
- `initialize_pinecone(self, api_key, pinecone_index_name)`: Initializes the Pinecone index.
- `setup_sub_agents(self, mode=str)`: Sets up the sub-agents with the specified mode.
- `main_loop(self, prompt)`: Main loop that coordinates the workflow.
- `prompt_with_checks(self, prompt, conv_history="")`: Processes the prompt through the filtering, querying, and answering agents.

## Usage

1. **Initialize the Head Agent**:
   ```python
   if "head_agent" not in st.session_state:
       st.session_state.head_agent = Head_Agent(get_openai_key(), get_pinecone_key(), pinecone_index_name=INDEX_NAME)
       st.session_state.head_agent.setup_sub_agents(mode=AgentStyle.PRECISE)
   ```

2. **Display existing chat messages**:
   ```python
   for message in st.session_state.messages:
       if message["role"] != "system":
           with st.chat_message(message["role"]):
               st.write(message["content"])
   ```

3. **Wait for user input and generate AI response**:
   ```python
   if prompt := st.chat_input("What would you like to chat about?"):
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
           st.write(prompt)
       with st.chat_message("assistant"):
           assistant_response = st.session_state.head_agent.main_loop(prompt)
           st.write(assistant_response)
       st.session_state.messages.append({"role": "assistant", "content": assistant_response})
   ```

4. **Select conversation mode**:
   ```python
   selected_mode = st.selectbox("Select Mode", options=[AgentStyle.CHATTY, AgentStyle.PRECISE], index=0)
   if selected_mode != st.session_state.head_agent.mode:
       st.session_state.head_agent.setup_sub_agents(mode=selected_mode)
   ```

This file documents the implementation and usage of the `qna.py` file for the clinical assistant chatbot.
