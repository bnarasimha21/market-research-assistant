# Market Research Assistant

A Streamlit-based web application that provides a beautiful visualization of market research between Narsi (A new AI tool), GitHub Copilot, and Cursor. You can change the product name to any product you want to compare.

## Features

- Interactive comparison table
- Downloadable CSV export
- Detailed category-wise comparison
- Responsive design

## Installation

1. Clone the repository:

``` bash
git clone https://github.com/bnarasimha/market-research-assistant.git
```

2. Create a virtual environment:

``` bash
python -m venv venv
```

3. Activate the virtual environment:

``` bash
source venv/bin/activate
```

4. Install required packages:

``` bash
pip install -r requirements.txt
```

5. Create GenAI Agents:
- Login to [DigitalOcean](https://cloud.digitalocean.com/login).
- For Product Research agent which is the first agent:
        - upload product_narsi.md file (available in the root directory) to spaces object storage.
        - create a knowledge base choosing the newly created spaces object storage as the source.
        - attach knowledge base to the agent. 

  To learn more about creating agents, refer to [GenAI Documentation](https://docs.digitalocean.com/products/genai-platform/how-to/manage-ai-agent/).
- Create an Agent each for Competitor Research, and Comparison Report.
- Agent instructions are available in the `agent_instructions.txt` file.
- Copy the endpoints and keys for each agent.


6. Set the environment variables:  

- Copy .env.example file and create a .env file in the root directory.
- Update the following variables from previous step:

``` bash
OPENAI_API_KEY=<your-openai-api-key>

GENAI_PRODUCT_RESEARCHER_AGENT_ENDPOINT=<your-agent-endpoint>
GENAI_PRODUCT_RESEARCHER_AGENT_KEY=<your-agent-key>

GENAI_COMPETITOR_RESEARCHER_AGENT_ENDPOINT=<your-agent-endpoint>
GENAI_COMPETITOR_RESEARCHER_AGENT_KEY=<your-agent-key>

GENAI_COMPARISON_REPORT_AGENT_ENDPOINT=<your-agent-endpoint>
GENAI_COMPARISON_REPORT_AGENT_KEY=<your-agent-key>
```

`OPENAI_API_KEY` is used by CrewAI's default LLM, which drives the three CrewAI agents. The `GENAI_*` endpoints and keys are used by the tools those agents call. Each tool sends the model name "DeepSeek R1 Distill Llama 70B" as a request parameter only; the model that actually answers is whatever you configured on the hosted GenAI agent behind that endpoint.

Product and competitor names are not known when the modules are imported. They are read from the Streamlit inputs when you click "Generate Comparison" and passed into the crews as inputs.

7. Run the application:

``` bash
streamlit run app.py
```

8. Open your browser and navigate to `http://localhost:8501`


## Project Structure

- `app.py`: Main Streamlit application file.
- `crew_flow.py`: The main logic for the market research assistant.
- `product_research.py`: The logic for the product research task.
- `competitor_research.py`: The logic for the competitor research task.
- `comparison_report.py`: The logic for the comparison report task.
- `requirements.txt`: List of dependencies.
- `README.md`: This file.
- `product_narsi.md`: The product information for Narsi.
- `agent_instructions.txt`: The instructions for the agents.
