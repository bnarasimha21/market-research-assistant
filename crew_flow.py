import os
import sys
from dotenv import load_dotenv
from crewai import Crew, Flow
from crewai.flow.flow import listen, start
from product_research import product_researcher, product_research_task
from competitor_research import competitor_analyst, competitor_research_task
from comparison_report import comparison_analyst, comparison_research_task
import streamlit as st

load_dotenv()


def _state(key: str) -> str:
    """Read a string input from Streamlit session state, always returning a str."""
    value = st.session_state.get(key)
    return value if isinstance(value, str) else ""


class MarketResearchAssistant(Flow):
    @start()
    def get_product_narsi_info(self):  
        product_research_crew = Crew(
            agents=[product_researcher], 
            tasks=[product_research_task]
        )

        my_product = _state("my_product")
        result = product_research_crew.kickoff(inputs={"my_product": my_product})
        return result
    
    @listen(get_product_narsi_info)
    def get_competitor_info(self):
        competitor_research_crew = Crew(
            agents=[competitor_analyst], 
            tasks=[competitor_research_task]
        )

        my_product = _state("my_product")
        competitor1 = _state("competitor1")
        competitor2 = _state("competitor2") 

        result = competitor_research_crew.kickoff(inputs={"my_product": my_product, "competitor1": competitor1, "competitor2": competitor2})
        return result

    @listen(get_competitor_info)
    def get_comparison_info(self):
        comparison_research_crew = Crew(
            agents=[comparison_analyst], 
            tasks=[comparison_research_task]
        )

        competitor1 = _state("competitor1")
        competitor2 = _state("competitor2")

        result = comparison_research_crew.kickoff(inputs={"competitor1": competitor1, "competitor2": competitor2})
        return result

if __name__ == "__main__":
    # CLI use: python crew_flow.py [my_product] [competitor1] [competitor2]
    # Values fall back to env vars, then to placeholder strings. The tools read
    # st.session_state, so seed it with the same strings.
    defaults = {"my_product": "MY_PRODUCT", "competitor1": "COMPETITOR1", "competitor2": "COMPETITOR2"}
    for idx, key in enumerate(defaults, start=1):
        value = sys.argv[idx] if len(sys.argv) > idx else os.getenv(key.upper(), defaults[key])
        st.session_state[key] = str(value)
    flow = MarketResearchAssistant()
    result = flow.kickoff()
    print(result)
