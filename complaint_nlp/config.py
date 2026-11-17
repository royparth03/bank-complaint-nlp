"""All tunable settings in one place."""

SEED = 42
DATE_FROM = "2018-01-01"     # after CFPB's 2017 product-taxonomy change
MIN_WORDS = 20               # narratives shorter than this carry too little signal
MIN_PRODUCT_VOLUME = 2_000   # products rarer than this are dropped
PER_PRODUCT = 2_500          # cap per product so credit reporting (65% of raw volume) doesn't swamp the rest
TEST_SHARE = 0.20
SEARCH_QUERIES = 2_000       # unseen complaints used as search queries
TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MAX_TOKENS = 128

# CFPB renamed some products over time; merge the renamed labels so each queue has one name
PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Payday loan, title loan, personal loan, or advance loan": "Payday, title or personal loan",
    "Payday loan, title loan, or personal loan": "Payday, title or personal loan",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
}

COLUMNS = ["Date received", "Product", "Issue", "Consumer complaint narrative", "Company",
           "Company response to consumer", "Timely response?"]
