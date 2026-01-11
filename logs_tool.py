from langchain_core.tools import tool
import httpx
import json
from typing import Optional

# New Relic API Configuration
NEW_RELIC_API_KEY = "NRAK-UL2MHU311Z8F954U31ZDJXBTA66"
NEW_RELIC_ACCOUNT_ID = 7527127
NEW_RELIC_GRAPHQL_URL = "https://api.newrelic.com/graphql"

@tool
def query_logs(
    query: str,
    account_id: Optional[int] = None,
    since_timestamp: Optional[int] = None
) -> str:
    """
    Use this tool to query logs.
    """
    try:
        # Use default account ID if not provided
        if account_id is None:
            account_id = NEW_RELIC_ACCOUNT_ID
        nrql_query = "SELECT timestamp, level, message FROM Log where message like '%{}%' SINCE {}"
        # Construct the full NRQL query with SINCE clause if timestamp is provided
        if since_timestamp is None:
            since_timestamp = 0
        full_query = nrql_query.format(query, since_timestamp)

        # Construct the GraphQL query
        graphql_query = f"""
                {{
                  actor {{
                    nrql(
                      accounts: {account_id}
                      query: "{full_query}"
                    ) {{
                      results
                    }}
                  }}
                }}
                """
        # Prepare headers
        required_headers = {
            "API-Key": NEW_RELIC_API_KEY,
            "Content-Type": "application/json"
        }

        client = httpx.Client(timeout=None, headers=required_headers)

        response = client.post(NEW_RELIC_GRAPHQL_URL, json={"query": graphql_query})
        response.raise_for_status()  # Check for 400, 500 errors
            
        # Parse the response
        result = response.json()

        # Extract the results from the GraphQL response
        if "data" in result and "actor" in result["data"]:
            nrql_data = result["data"]["actor"].get("nrql", {})
            results = nrql_data.get("results", [])

            if not results:
                return "No log entries found matching the query."

            # Format the results as a readable string
            formatted_results = "New Relic Log Query Results:\n"
            formatted_results += "=" * 50 + "\n\n"

            for i, entry in enumerate(results, 1):
                formatted_results += f"Entry #{i}:\n"
                for key, value in entry.items():
                    formatted_results += f"  {key}: {value}\n"
                formatted_results += "\n"

            return formatted_results
        else:
            # Check for errors in the response
            if "errors" in result:
                error_msg = json.dumps(result["errors"], indent=2)
                return f"GraphQL Error: {error_msg}"
            return f"Unexpected response format: {json.dumps(result, indent=2)}"
                
    except httpx.HTTPStatusError as e:
        return f"HTTP Error {e.response.status_code}: {e.response.text}"
    except httpx.RequestError as e:
        return f"Request Error: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON Decode Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"