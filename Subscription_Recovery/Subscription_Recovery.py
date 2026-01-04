import streamlit as st
import pandas as pd
import os
import tempfile
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# ============================================================================
# 1. CORE LOGIC (The "Math" Tool)
# ============================================================================

def find_subscription_leeches(csv_file):
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['merchant', 'date'])
    # Identify recurring merchants
    recurring = df[df.duplicated('merchant', keep=False)].copy()
    # Calculate price change
    recurring['price_change'] = recurring.groupby('merchant')['amount'].diff()
    # Return rows where price increased
    return recurring[recurring['price_change'] > 0]

class SubscriptionTools:
    @tool("analyze_transactions")
    def analyze_transactions(csv_path: str):
        """Useful to identify recurring payments and price increases from a CSV."""
        try:
            leeches = find_subscription_leeches(csv_path)
            if leeches.empty:
                return "No price hikes found."
            return leeches.to_json(orient='records') # JSON is easier for LLMs to read
        except Exception as e:
            return f"Error: {str(e)}"

# ============================================================================
# 2. STREAMLIT UI SETUP
# ============================================================================

st.set_page_config(page_title="Subscription Leech Recovery", page_icon="🧛")
st.title("🧛 Subscription Leech Recovery")
st.markdown("Automated forensic accounting and negotiation for your subscriptions.")

# Sidebar for API Configuration
with st.sidebar:
    st.header("API Configuration")
    cerebras_key = st.secrets.get("CEREBRAS_API_KEY") or st.text_input("Cerebras API Key", type="password")
    
    if not cerebras_key:
        st.warning("Please add your Cerebras Key in Secrets or Sidebar.")

# ============================================================================
# 3. FILE UPLOADER & EXECUTION
# ============================================================================

uploaded_file = st.file_uploader("Upload your transaction CSV", type="csv")

if uploaded_file and cerebras_key:
    # Save upload to a temp file for the Tool to access
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if st.button("🔍 Find & Recover My Money"):
        try:
            with st.status("🧠 Initializing Agents...", expanded=True) as status:
                
                # Setup the LLM
                cerebras_llm = LLM(
                    model="cerebras/llama-3.3-70b",
                    api_key=cerebras_key,
                    base_url="https://api.cerebras.ai/v1",
                    temperature=0
                )

                # Define Agents
                accountant = Agent(
                    role='Forensic Financial Accountant',
                    goal='Quantify subscription price increases from CSV data',
                    backstory='You are a meticulous auditor who spots sneaky price hikes.',
                    tools=[SubscriptionTools.analyze_transactions],
                    llm=cerebras_llm,
                    verbose=True
                )

                negotiator = Agent(
                    role='Expert Consumer Advocate',
                    goal='Draft persuasive scripts to get refunds or lower rates',
                    backstory='You are a master of customer retention psychology.',
                    llm=cerebras_llm,
                    verbose=True
                )

                # Define Tasks
                t1 = Task(
                    description=f"Analyze the CSV at {tmp_path} and list all price hikes.",
                    expected_output="A list of merchants with their old and new prices.",
                    agent=accountant
                )

                t2 = Task(
                    description="Take the findings and write a 3-step negotiation email for the largest hike.",
                    expected_output="A full, professional negotiation script.Ensure the final script uses proper spacing, especially between numbers, currencies, and the start of new sentences",
                    agent=negotiator,
                    context=[t1]
                )

                # Run Crew
                st.write("🕵️ Searching for leeches...")
                crew = Crew(agents=[accountant, negotiator], tasks=[t1, t2], verbose=True)
                result = crew.kickoff()
                
                status.update(label="✅ Analysis Complete!", state="complete")

            # Display Results
            st.subheader("📋 Final Negotiation Script")
            st.markdown(result.raw)
            
            st.download_button("💾 Download Script", result.raw, "negotiation.txt")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
else:

    st.info("💡 Tip: Your CSV should have 'date', 'merchant', and 'amount' columns.")
