import streamlit as st
from langchain_core.messages import HumanMessage
from agent import create_graph

st.set_page_config(page_title="Rishabh's Dissertation Project")

st.title("Customer Support Chat Agent")

# Initialize graph if not in session state
if "graph" not in st.session_state:
    try:
        # Load the graph (this executes the time-consuming tool loading)
        st.session_state.graph = create_graph()
        st.success("Agent graph initialized with tools.")
    except Exception as e:
        st.error(f"Failed to initialize agent graph. Is the MCP server running? Error: {e}")
        st.session_state.graph = None # Prevent subsequent runs

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Messages (Always runs on page load) ---
for msg in st.session_state.messages:

    # Check if the message is a user message OR an AI message with final content
    is_user_message = msg.type == "human"
    is_final_ai_response = msg.type == "ai" and msg.content and not msg.tool_calls

    # We will ONLY display the HumanMessage and the final, content-filled AIMessage
    if is_user_message or is_final_ai_response:
        # Determine the role for the chat UI
        role = "user" if msg.type == "human" else "assistant"

        # Use st.chat_message for a nicer UI
        with st.chat_message(role):
            # Display the content
            st.markdown(msg.content)

# --- Chat Input / Run Logic ---
if st.session_state.graph:
    # Use the chat-friendly input widget
    prompt = st.chat_input("Ask a question about employees...")

    if prompt:
        # 1. Append the user message to the display list
        st.session_state.messages.append(HumanMessage(content=prompt))

        # 2. Invoke the graph
        # This will run the agent logic based on the *updated* state
        with st.spinner("Thinking..."):
            result = st.session_state.graph.invoke(
                {"messages": st.session_state.messages}
            )

        # 3. Update the session state with the final result messages
        st.session_state.messages = result["messages"]

        # Rerun the Streamlit app to update the display with new messages
        st.rerun()