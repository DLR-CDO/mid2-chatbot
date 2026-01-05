import os
import csv
import json
import asyncio
import re
from typing import List, Dict, Any, Annotated, Optional
import pyodbc
from filter_resolver import *
import chainlit as cl
# from nl2sql import NL2SQLChatbot

from dotenv import load_dotenv

import pandas as pd
import json
import base64
import httpx
from openai import AzureOpenAI
from datetime import datetime

# Semantic Kernel imports
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()

# Updated to match your new connection string format
DB_SERVER = os.getenv("DB_Data_Source")
DB_DATABASE = os.getenv("DB_Initial_Catalog")
DB_USER = os.getenv("DB_User_ID")
DB_PASSWORD = os.getenv("DB_Password")

# ============================================================
# LOAD METADATA + DESCRIPTIONS
# ============================================================

def load_prompt(file_path: str) -> str:
    """Load prompt from text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_table_descriptions():
    """Load table descriptions from CSV."""
    with open("data/table_description.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_metadata():
    """Load table metadata (columns, synonyms, metrics)."""
    with open("data/metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

table_desc = load_table_descriptions()
metadata = load_metadata()

# ============================================================
# SEMANTIC KERNEL SETUP
# ============================================================

class NL2SQLChatbot:
    def __init__(self):
        self.kernel = sk.Kernel()
        
        # Initialize Azure OpenAI service
        self.chat_service = AzureChatCompletion(
            endpoint=os.getenv("AZURE_OPENAI_BASE_URL"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        self.kernel.add_service(self.chat_service)
        self.chat_service = self.kernel.get_service()
        
        # Initialize plugins
        self._setup_plugins()

        # print("\n=== TESTING REGISTERED FUNCTIONS ===")
        # try:
        #     funcs = self.kernel.get_plugin("nl2sql")
        #     # print(f"Registered functions in nl2sql plugin: {list(funcs.keys())}")
        #     print(funcs)
        # except Exception as e:
        #     print(f"Error getting functions: {e}")
        # print("=== END TEST ===\n")
    
    def analyze_filters(self, sql_query: str, relevant_tables: List[str]) -> Dict[str, Any]:
        """Analyze which filters are applied in the SQL query."""
        
        filter_info = {
            "operational_status": {
                "present_in_tables": False,
                "column_exists": False,
                "filters_applied": [],
                "possible_values": []
            },
            "property_type": {
                "present_in_tables": False,
                "column_exists": False,
                "filters_applied": [],
                "possible_values": []
            }
        }
        
        # 1. Check if columns exist in metadata for relevant tables
        for table in relevant_tables:
            table_metadata = next((item for item in metadata if item.get("table") == table), None)
            if table_metadata:
                columns = table_metadata.get("columns", [])
                column_names = [col.get("name", "").lower() for col in columns]
                column_synonyms = []
                
                # Get all synonyms
                for col in columns:
                    synonyms = col.get("synonyms", [])
                    column_synonyms.extend([s.lower() for s in synonyms])
                
                # Check for operationalstatus
                if "operationalstatus" in column_names or "operationalstatus" in column_synonyms:
                    filter_info["operational_status"]["column_exists"] = True
                    filter_info["operational_status"]["present_in_tables"] = True
                    
                    # Try to get possible values from column metadata
                    for col in columns:
                        if col.get("name", "").lower() == "operationalstatus":
                            values = col.get("values", [])
                            filter_info["operational_status"]["possible_values"] = values
                            break
                
                # Check for propertytype
                if "propertytype" in column_names or "propertytype" in column_synonyms:
                    filter_info["property_type"]["column_exists"] = True
                    filter_info["property_type"]["present_in_tables"] = True
                    
                    # Try to get possible values from column metadata
                    for col in columns:
                        if col.get("name", "").lower() == "propertytype":
                            values = col.get("values", [])
                            filter_info["property_type"]["possible_values"] = values
                            break
        
        # 2. Parse SQL query for applied filters
        sql_lower = sql_query.lower()
        
        # Check for OperationalStatus filters
        if "operationalstatus" in sql_lower:
            # Look for specific patterns
            import re
            
            # Pattern for OperationalStatus = 'value'
            op_status_pattern = r"operationalstatus\s*=\s*'([^']+)'"
            op_status_in_pattern = r"operationalstatus\s+in\s*\(([^)]+)\)"
            
            # Find exact matches
            exact_matches = re.findall(op_status_pattern, sql_lower)
            # Find IN clause matches
            in_matches = re.findall(op_status_in_pattern, sql_lower)
            
            # Collect all values
            all_values = []
            if exact_matches:
                all_values.extend([match.upper() for match in exact_matches])
            if in_matches:
                values_str = in_matches[0]
                values = re.findall(r"'([^']+)'", values_str)
                all_values.extend([v.upper() for v in values])
            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            for value in all_values:
                if value not in seen:
                    seen.add(value)
                    unique_values.append(value)
            
            if unique_values:
                filter_info["operational_status"]["filters_applied"] = unique_values
            
            # Check for inequality operators
            if not filter_info["operational_status"]["filters_applied"]:
                if "operationalstatus !=" in sql_lower or "operationalstatus <>" in sql_lower:
                    filter_info["operational_status"]["filters_applied"] = ["FILTERED_OUT"]
        
        # Check for PropertyType filters
        if "propertytype" in sql_lower:
            import re
            
            # Pattern for PropertyType = 'value'
            prop_type_pattern = r"propertytype\s*=\s*'([^']+)'"
            prop_type_in_pattern = r"propertytype\s+in\s*\(([^)]+)\)"
            
            # Find exact matches
            exact_matches = re.findall(prop_type_pattern, sql_lower)
            # Find IN clause matches
            in_matches = re.findall(prop_type_in_pattern, sql_lower)
            
            # Collect all values
            all_values = []
            if exact_matches:
                all_values.extend([match.upper() for match in exact_matches])
            if in_matches:
                values_str = in_matches[0]
                values = re.findall(r"'([^']+)'", values_str)
                all_values.extend([v.upper() for v in values])
                
            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            for value in all_values:
                if value not in seen:
                    seen.add(value)
                    unique_values.append(value)
            
            if unique_values:
                filter_info["property_type"]["filters_applied"] = unique_values
            
            # Check for inequality operators (if needed)
            if not filter_info["property_type"]["filters_applied"]:
                if "propertytype !=" in sql_lower or "propertytype <>" in sql_lower:
                    filter_info["property_type"]["filters_applied"] = ["FILTERED_OUT"]
        
        return filter_info

    def format_filter_info(self, filter_info: Dict[str, Any]) -> str:
        """Format filter information for display (simplified version)."""
        
        display_parts = []
        
        # Check if OperationalStatus is present and add simple display
        if filter_info.get("operational_status", {}).get("present_in_tables"):
            if filter_info["operational_status"].get("filters_applied"):
                filters = ", ".join(filter_info["operational_status"]["filters_applied"])
                display_parts.append(f"**Operational Status:** {filters}")
            else:
                display_parts.append("**Operational Status:** ALL VALUES")
        
        # Check if PropertyType is present and add simple display
        if filter_info.get("property_type", {}).get("present_in_tables"):
            if filter_info["property_type"].get("filters_applied"):
                filters = ", ".join(filter_info["property_type"]["filters_applied"])
                display_parts.append(f"**Property Type:** {filters}")
            else:
                display_parts.append("**Property Type:** ALL VALUES")
        
        if display_parts:
            return "\n".join(display_parts)
        return ""
        
    def _setup_plugins(self):
        """Setup Semantic Kernel plugins."""
        # Create table selection function
        @kernel_function(description="Select relevant tables for a natural language query",name="pick_relevant_tables")
        async def pick_relevant_tables(nl_query: Annotated[str, "The natural language query from user"]
        ) -> Annotated[List[str], "List of relevant table names"]:
            """Select relevant tables using LLM with context."""
            
            # 1. Load system prompt from external file
            system_prompt = load_prompt("prompts/tableSelection.txt")
            
            # 2. Prepare metadata for assistant message (JSON format like Groq example)
            table_list = [
                {"table_name": row.get("table_name"), "description": row.get("description")}
                for row in table_desc
            ]
            
            calc_info = [
                {
                    "table": item.get("table"),
                    "calculatedMetrics": item.get("calculatedMetrics", [])
                }
                for item in metadata if item.get("calculatedMetrics")
            ]
            
            metadata_content = json.dumps({
                "tableDescriptions": table_list,
                "calculatedMetrics": calc_info
            }, indent=2)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": metadata_content},
                {"role": "user", "content": nl_query}
            ]
            
            # 4. Convert to Semantic Kernel ChatHistory
            chat_history = ChatHistory()
            for msg in messages:
                chat_history.add_message(ChatMessageContent(
                    role=msg["role"],
                    content=msg["content"]
                ))
            
            execution_settings = PromptExecutionSettings(
                temperature=0.0,
                max_tokens=1000,
            )
            
            result = await self.chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=execution_settings
            )
            
            response = result[0].content
            
            try:
                return json.loads(response)
            except:
                return []
        
        # Create SQL generation function
        @kernel_function(
        description="Generate SQL query for natural language question",name="generate_sql")
        async def generate_sql(
            nl_query: Annotated[str, "The natural language query"],
            relevant_tables: Annotated[List[str], "List of relevant table names from previous step"],
            resolved_filters: Annotated[Dict[str, str], "Resolved filter values"],
            chat_history: Annotated[ChatHistory, "Chat history for context"] = None
        ) -> Annotated[Dict[str, str], "SQL query and description"]:
            """Generate SQL using Azure OpenAI LLM with filtered metadata."""
            
            # 1. Load system prompt from external file
            system_prompt = load_prompt("prompts/sqlGeneration.txt")
            
            # 2. Filter metadata for selected tables
            filtered_metadata = [
                item for item in metadata
                if item.get("table") in relevant_tables
                or item.get("procedure") in relevant_tables
            ]
            resolved_filter_block = f"""
                ### RESOLVED FILTER VALUES (DO NOT MODIFY) ###
                {json.dumps(resolved_filters, indent=2)}
            """

            # 3. Prepare metadata for assistant message
            metadata_content = (
                "### METADATA START ###\n"
                + json.dumps(filtered_metadata, indent=2)
                + "\n"
                + resolved_filter_block
                + "\n### METADATA END ###"
            )
            
            # 4. Build the 3-message pattern for Azure OpenAI
            new_chat_history = ChatHistory()
            new_chat_history.add_system_message(system_prompt)
            new_chat_history.add_assistant_message(metadata_content)
            new_chat_history.add_user_message(nl_query)
            
            # 5. Add conversation history if provided
            if chat_history and len(chat_history.messages) > 0:
                for msg in chat_history.messages[-4:]:  # Last 4 messages for context
                    new_chat_history.add_message(msg)
            
            # 6. Azure OpenAI execution settings
            execution_settings = PromptExecutionSettings(
                temperature=0.0,
                max_tokens=2000,
            )
            
            # 7. Call Azure OpenAI via Semantic Kernel
            result = await self.chat_service.get_chat_message_contents(
                chat_history=new_chat_history,
                settings=execution_settings
            )
            
            response = result[0].content
            
            try:
                return json.loads(response)
            except:
                return {
                    "description": "Generated SQL query",
                    "sql": response
                }
        
        # Create database execution function
        @kernel_function(description="Execute SQL query on Azure SQL database",name="run_sql")
        async def run_sql(
            sql_query: Annotated[str, "SQL query to execute"]
        ) -> Annotated[List[Dict], "Query results"]:
            """Execute SQL query on Azure SQL Server using SQL authentication."""
            
            try:
                # Build connection string for SQL Server authentication
                conn_str = (
                    "DRIVER={ODBC Driver 18 for SQL Server};"
                    f"SERVER={DB_SERVER};"
                    f"DATABASE={DB_DATABASE};"
                    f"UID={DB_USER};"
                    f"PWD={DB_PASSWORD};"
                    "Encrypt=yes;"
                    "TrustServerCertificate=no;"
                    "Connection Timeout=30;"
                )
                
                print(f"Connecting to: {DB_SERVER}, Database: {DB_DATABASE}")
                
                # Connect using SQL Server authentication
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                
                cursor.execute(sql_query)
                
                # Get column names
                columns = [column[0] for column in cursor.description]
                
                # Fetch all rows and convert to list of dictionaries
                rows = cursor.fetchall()
                results = []
                
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        row_dict[col] = row[i]
                    results.append(row_dict)
                
                cursor.close()
                conn.close()
                
                return results
                
            except Exception as e:
                return [{"error": str(e)}]
            
        @kernel_function(
            description="Generate follow-up question suggestions based on query context",
            name="generate_suggestions"
        )
        async def generate_suggestions(
            nl_query: Annotated[str, "The original natural language query"],
            relevant_tables: Annotated[List[str], "Tables used in the query"],
            sql_query: Annotated[str, "The SQL query that was executed"],
            query_results: Annotated[Any, "The query results (any format)"],
            chat_history: Annotated[ChatHistory, "Chat history for context"] = None
        ) -> Annotated[List[str], "List of suggested follow-up questions"]:
            """Generate context-aware follow-up question suggestions."""
            
            print(f"\n=== START GENERATE_SUGGESTIONS ===")
            
            # SIMPLER APPROACH: Don't pass query_results to LLM at all
            # Just use the query and table info
            
            # Simple prompt
            prompt = f"""The user asked this question about database tables: "{nl_query}"
                    
                The query involved these tables: {', '.join(relevant_tables)}

                The SQL query was: {sql_query[:150]}...

                Based on this, suggest 3 specific follow-up questions the user might ask next.
                Make each question explore a different angle (trends, comparisons, breakdowns, etc.).
                Format as a simple JSON array: ["question 1?", "question 2?", "question 3?"]"""
            
            # Build messages
            new_chat_history = ChatHistory()
            new_chat_history.add_user_message(prompt)
            
            # Execution settings
            execution_settings = PromptExecutionSettings(
                temperature=0.7,
                max_tokens=300,
            )
            
            # Call LLM
            try:
                result = await self.chat_service.get_chat_message_contents(
                    chat_history=new_chat_history,
                    settings=execution_settings
                )
                
                response = result[0].content
                print(f"DEBUG: LLM response: {response}")
                
                # Parse response - try to extract JSON
                try:
                    # Find JSON array in response
                    import re
                    json_match = re.search(r'\[.*\]', response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        if isinstance(data, list):
                            suggestions = [str(s).strip() for s in data[:3]]
                            # Ensure questions end with ?
                            suggestions = [s if s.endswith('?') else s.rstrip('.') + '?' for s in suggestions]
                            print(f"DEBUG: Parsed suggestions: {suggestions}")
                            return suggestions
                except:
                    pass
                
                print(f"DEBUG: Using fallback suggestions")
                return []
                
            except Exception as e:
                pass
                # print(f"DEBUG: Error: {e}")
                return [] 
            
        # Add all functions to kernel
        self.kernel.add_function(
            plugin_name="nl2sql",
            function=pick_relevant_tables
        )
        
        self.kernel.add_function(
            plugin_name="nl2sql",
            function=generate_sql
        )
        
        self.kernel.add_function(
            plugin_name="nl2sql",
            function=run_sql
        )
        
        self.kernel.add_function(
            plugin_name="nl2sql",
            function=generate_suggestions
        )

        self.kernel.add_function(
            plugin_name="filters",
            function=resolve_filter_values
        )

    
    async def process_query(self, nl_query: str, chat_history: ChatHistory = None) -> Dict[str, Any]:
        """Process natural language query through the full pipeline."""
        
        print(f"\n=== START PROCESS_QUERY for: '{nl_query}' ===")
        
        # Step 1: Table selection
        pick_tables_func = self.kernel.get_function(
            plugin_name="nl2sql",
            function_name="pick_relevant_tables"
        )
        
        args = KernelArguments(nl_query=nl_query)
        relevant_tables = await self.kernel.invoke(pick_tables_func, arguments=args)
        print(f"DEBUG: Relevant tables result : {relevant_tables}")
        print(f"DEBUG: Relevant tables value: {relevant_tables.value}")
        relevant_tables = relevant_tables.value
        
        if not relevant_tables:
            print("DEBUG: No relevant tables found, returning early")
            return {
                "relevant_tables": [],
                "sql": None,
                "results": None,
                "suggestions": [],
                "error": "No tables match the question. SQL generation skipped."
            }

        # STEP 1.5: Resolve filter values using AI Search
        resolve_filters_func = self.kernel.get_function(
            plugin_name="filters",
            function_name="resolve_filter_values"
        )

        args = KernelArguments(nl_query=nl_query)
        resolved_filters_result = await self.kernel.invoke(resolve_filters_func, arguments=args)
        resolved_filters = resolved_filters_result.value or {}

        print(f"DEBUG: Resolved filters from AI Search: {resolved_filters}")
        
        # Step 2: SQL generation with chat history
        generate_sql_func = self.kernel.get_function(
            plugin_name="nl2sql",
            function_name="generate_sql"
        )
        
        args = KernelArguments(
            nl_query=nl_query,
            relevant_tables=relevant_tables,
            chat_history=chat_history,
            resolved_filters=resolved_filters 
        )
        
        sql_result = await self.kernel.invoke(generate_sql_func, arguments=args)
        print(f"DEBUG: SQL result: {sql_result}")
        sql_data = sql_result.value
        
        if isinstance(sql_data, dict):
            sql_query = sql_data.get("sql")
            description = sql_data.get("description")
        else:
            sql_query = sql_data
            description = "Generated SQL query"
        
        print(f"DEBUG: Generated SQL: {sql_query}...")
        
        # Step 2.5: Analyze applied filters BEFORE execution
        filter_info = self.analyze_filters(sql_query, relevant_tables)

        # Step 3: Execute SQL
        run_sql_func = self.kernel.get_function(
            plugin_name="nl2sql",
            function_name="run_sql"
        )
        
        args = KernelArguments(sql_query=sql_query)
        results = await self.kernel.invoke(run_sql_func, arguments=args)
        query_results = results.value
        
        print(f"DEBUG: Query results : {query_results}")
        print(f"DEBUG: Query results length: {len(query_results) if isinstance(query_results, list) else 'N/A'}")
        
        # Step 4: Generate suggestions (only if we have valid results)
        suggestions = []
        if query_results and isinstance(query_results, list):
            # Check if it's actual results (not an error)
            has_valid_results = (len(query_results) > 0 and 
                isinstance(query_results[0], dict) and "error" not in query_results[0]
            )
            
            print(f"DEBUG: Has valid results: {has_valid_results}")
            
            if has_valid_results:
                try:
                    generate_suggestions_func = self.kernel.get_function(
                        plugin_name="nl2sql",
                        function_name="generate_suggestions"
                    )
                    print(f"DEBUG: Got suggestions function")
                    
                    args = KernelArguments(
                        nl_query=nl_query,
                        relevant_tables=relevant_tables,
                        sql_query=sql_query,
                        query_results=query_results,
                        chat_history=chat_history
                    )
                    
                    print(f"DEBUG: Invoking suggestions function...")
                    suggestions_task = asyncio.create_task(
                        self.kernel.invoke(generate_suggestions_func, arguments=args)
                    )
                    suggestions_result = await asyncio.wait_for(suggestions_task, timeout=3.0)
                    print(f"DEBUG: Suggestions result: {suggestions_result}")
                    suggestions = suggestions_result.value if suggestions_result.value else []
                    print(f"DEBUG: Generated suggestions: {suggestions}")
                    
                except asyncio.TimeoutError:
                    print("DEBUG: Suggestions generation timed out")
                except Exception as e:
                    print(f"DEBUG: Suggestions generation error: {str(e)}")
        else:
            print(f"DEBUG: Skipping suggestions - invalid query results")
        
        print(f"=== END PROCESS_QUERY ===")
        
        return {
            "relevant_tables": relevant_tables,
            "description": description,
            "sql": sql_query,
            "results": query_results,
            "suggestions": suggestions,
            "filter_info": filter_info
        }
    
    # In nl2sql.py, replace render_chart method with:
    async def prepare_chart_data(
        self,
        results: List[Dict[str, Any]],
        user_query: str
    ) -> Dict[str, Any]:

        if not results or len(results) < 2:
            return {"error": "Not enough data to visualize."}

        df = pd.DataFrame(results)
        columns = df.columns.tolist()

        # Build strict data summary
        summary_lines = []
        for col in columns:
            example = df[col].iloc[0]
            dtype = str(df[col].dtype)
            summary_lines.append(f"{col} ({dtype}) → example: {example}")

        data_summary = "\n".join(summary_lines)

        system_prompt = load_prompt("prompts/visualGeneration.txt")

        prompt_filled = system_prompt \
            .replace("{{USER_QUERY}}", user_query) \
            .replace("{{COLUMNS}}", ", ".join(columns)) \
            .replace("{{DATA_SUMMARY}}", data_summary)

        # Call LLM (chat model, NOT image model)
        execution_settings = PromptExecutionSettings(
            temperature=0.0,
            max_tokens=500
        )

        chat_history = ChatHistory()
        chat_history.add_system_message(prompt_filled)

        result = await self.chat_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings
        )

        try:
            chart_spec = json.loads(result[0].content)
        except Exception:
            return {"error": "Invalid visualization spec generated."}

        if "error" in chart_spec:
            return chart_spec

        return {
            "chart_spec": chart_spec,
            "rows": len(df),
            "columns": columns
        }


    async def generate_chart_image(self, image_prompt: str) -> Dict[str, Any]:
        """Generate chart image using Azure OpenAI Image API."""
        
        try:
            # 5️⃣ Azure Image API config
            image_endpoint = os.getenv(
                "AZURE_IMAGE_ENDPOINT",
                "https://sfdevopenaidel.openai.azure.com"
            ).rstrip("/")

            image_api_key = os.getenv("AZURE_IMAGE_API_KEY") or os.getenv("AZURE_API_KEY")
            image_api_version = os.getenv("AZURE_IMAGE_API_VERSION", "2024-02-01")
            image_deployment = os.getenv("AZURE_IMAGE_MODEL", "gpt-image-1-mini")

            if not image_api_key:
                return {"error": "AZURE_IMAGE_API_KEY not set."}

            image_url = (
                f"{image_endpoint}/openai/deployments/"
                f"{image_deployment}/images/generations"
                f"?api-version={image_api_version}"
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {image_api_key}",
            }

            payload = {
                "prompt": image_prompt,
                "size": "1024x768",
                "quality": "medium",
                "background": "auto",
                "output_format": "png",
                "n": 1
            }

            # 6️⃣ Call image generation
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(image_url, headers=headers, json=payload)

            response.raise_for_status()
            result = response.json()

            base64_image = result["data"][0]["b64_json"]
            if not base64_image:
                return {"error": "No image returned by the model."}

            return {"base64_image": base64_image, "success": True}
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return {"error": str(e)}