import os
from typing import Dict, Annotated
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from semantic_kernel.functions.kernel_function_decorator import kernel_function

# Azure AI Search config
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)

@kernel_function(
    name="resolve_filter_values",
    description="Resolve canonical dimension values using Azure AI Search"
)
async def resolve_filter_values(
    nl_query: Annotated[str, "User natural language query"]
) -> Annotated[Dict[str, str], "Resolved filter values"]:
    """
    Uses Azure AI Search to ground filter values like Market, Country, City, Provider
    """
    resolved_filters = {}

    # ---- MARKET RESOLUTION ----
    market_results = search_client.search(
        search_text=nl_query,
        select=["Market"],
        top=3
    )

    for r in market_results:
        if r.get("Market"):
            resolved_filters["Market"] = r["Market"]
            break

    # ---- COUNTRY RESOLUTION ----
    country_results = search_client.search(
        search_text=nl_query,
        select=["Country"],
        top=3
    )

    for r in country_results:
        if r.get("Country"):
            resolved_filters["Country"] = r["Country"]
            break

    # ---- CITY RESOLUTION (optional) ----
    city_results = search_client.search(
        search_text=nl_query,
        select=["City"],
        top=3
    )

    for r in city_results:
        if r.get("City"):
            resolved_filters["City"] = r["City"]
            break

    # ---- GlobalRegion RESOLUTION ----
    global_region_results = search_client.search(
        search_text=nl_query,
        select=["GlobalRegion"],
        top=3
    )

    for r in global_region_results:
        if r.get("GlobalRegion"):
            resolved_filters["GlobalRegion"] = r["GlobalRegion"]
            break

    # ---- BusinessRegion RESOLUTION ----
    business_region_results = search_client.search(
        search_text=nl_query,
        select=["BusinessRegion"],
        top=3
    )

    for r in business_region_results:
        if r.get("BusinessRegion"):
            resolved_filters["BusinessRegion"] = r["BusinessRegion"]
            break

    # ---- CustomerType RESOLUTION ----
    customer_type_results = search_client.search(
        search_text=nl_query,
        select=["CustomerType"],
        top=3
    )

    for r in customer_type_results:
        if r.get("CustomerType"):
            resolved_filters["CustomerType"] = r["CustomerType"]
            break
    
    # ---- PropertyType RESOLUTION ----
    property_type_results = search_client.search(
        search_text=nl_query,
        select=["PropertyType"],
        top=3
    )

    for r in property_type_results:
        if r.get("PropertyType"):
            resolved_filters["PropertyType"] = r["PropertyType"]
            break

    return resolved_filters
