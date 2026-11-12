"""The fixed comparison category list.

This used to live only as prose in agent_instructions.txt and was never read
by any code, so the model was free to invent its own categories. Code now
reads this list; agent_instructions.txt is kept in sync with it manually
(see the comment there).
"""

from __future__ import annotations

CATEGORIES: tuple[str, ...] = (
    "Intelligent Code Generation",
    "Integration Capabilities",
    "Natural Language Interface",
    "User Experience",
    "Pricing and Licensing",
    "Community and Support",
    "Security and Privacy",
    "Scalability and Performance",
    "Documentation and Learning Resources",
    "Future Development and Roadmap",
)

MAX_CELL_LENGTH = 600
