import os

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
STM_CONDENSE_THRESHOLD = int(os.getenv("STM_CONDENSE_THRESHOLD", 5))

# Database URL will be provided by Render
DATABASE_URL = os.getenv("DATABASE_URL")
