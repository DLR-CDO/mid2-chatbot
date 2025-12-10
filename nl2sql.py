import os
import csv
import json
import asyncio
from typing import List, Dict, Any, Annotated, Optional
import pyodbc
import struct

from dotenv import load_dotenv
import chainlit as cl
from chainlit.message import Message
from azure.identity import ClientSecretCredential

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

SQL_ENDPOINT = os.getenv("SQL_ENDPOINT")
DATABASE = os.getenv("DATABASE")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

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
        print(os.getenv("AZURE_OPENAI_BASE_URL"))
        print(os.getenv("AZURE_OPENAI_API_KEY"))
        print(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
        print(os.getenv("AZURE_OPENAI_API_VERSION"))

        # chat_completion = 
        self.kernel.add_service(self.chat_service)
        self.chat_service = self.kernel.get_service()
        # Initialize plugins
        self._setup_plugins()
        
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
            
            # 3. Build the 3-message pattern exactly like Groq
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
            
            # 3. Prepare metadata for assistant message
            metadata_content = "### METADATA START ###\n" + \
                            json.dumps(filtered_metadata, indent=2) + \
                            "\n### METADATA END ###"
            
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
        
        def get_access_token() -> str:
            """Retrieve Azure AD token for SQL Authentication."""
            credential = ClientSecretCredential(
                tenant_id=os.getenv("TENANT_ID"),
                client_id=os.getenv("CLIENT_ID"),
                client_secret=os.getenv("CLIENT_SECRET")
            )
            token = credential.get_token(os.getenv("SCOPE"))
            return token.token

        
        # Create database execution function
        @kernel_function(description="Execute SQL query on Azure SQL database",name="run_sql")
        async def run_sql(
            sql_query: Annotated[str, "SQL query to execute"]
        ) -> Annotated[List[Dict], "Query results"]:
            """Execute SQL query on Azure SQL Server."""
            
            print("\nAuthenticating with Azure AD...")
            access_token = get_access_token()

            token_bytes = access_token.encode("utf-16-le")
            token_struct = struct.pack("=i", len(token_bytes)) + token_bytes

            attrs_before = {1256: token_struct}  # Access token attribute

            conn_str = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                f"SERVER=tcp:{SQL_ENDPOINT},1433;"
                f"DATABASE={DATABASE};"
                "Encrypt=yes;"
                "TrustServerCertificate=no;"
                "Connection Timeout=30;"
            )
            
            try:
                conn = pyodbc.connect(conn_str,attrs_before=attrs_before)
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
        
        # Add functions to kernel
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
    
    async def process_query(self, nl_query: str, chat_history: ChatHistory = None) -> Dict[str, Any]:
        """Process natural language query through the full pipeline."""
        
        # Step 1: Table selection
        pick_tables_func = self.kernel.get_function(
            plugin_name="nl2sql",
            function_name="pick_relevant_tables"
        )
        
        args = KernelArguments(nl_query=nl_query)
        relevant_tables = await self.kernel.invoke(pick_tables_func, arguments=args)
        relevant_tables = relevant_tables.value
        
        if not relevant_tables:
            return {
                "relevant_tables": [],
                "sql": None,
                "results": None,
                "error": "No tables match the question. SQL generation skipped."
            }
        
        # Step 2: SQL generation with chat history
        generate_sql_func = self.kernel.get_function(
            plugin_name="nl2sql",
            function_name="generate_sql"
        )
        
        args = KernelArguments(
            nl_query=nl_query,
            relevant_tables=relevant_tables,
            chat_history=chat_history
        )
        
        sql_result = await self.kernel.invoke(generate_sql_func, arguments=args)
        sql_data = sql_result.value
        
        if isinstance(sql_data, dict):
            sql_query = sql_data.get("sql")
            description = sql_data.get("description")
        else:
            sql_query = sql_data
            description = "Generated SQL query"
        
        # Step 3: Execute SQL
        run_sql_func = self.kernel.get_function(
            plugin_name="nl2sql",
            function_name="run_sql"
        )
        
        args = KernelArguments(sql_query=sql_query)
        results = await self.kernel.invoke(run_sql_func, arguments=args)
        
        return {
            "relevant_tables": relevant_tables,
            "description": description,
            "sql": sql_query,
            "results": results.value
        }

# ============================================================
# CHAINLIT APPLICATION
# ============================================================

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  return default_user

@cl.on_chat_start
async def on_chat_start():
    """Initialize chatbot when chat starts."""
    chatbot = NL2SQLChatbot()
    print('user instance :',chatbot)
    cl.user_session.set("chatbot", chatbot)
    
    # Initialize chat history
    chat_history = ChatHistory()
    cl.user_session.set("chat_history", chat_history)
    
    # Send welcome message
    welcome_msg = """🚀 **NL2SQL Chatbot**
    
        I can help you query your database using natural language. Just ask questions like:
        - "Show me total power capacity by region"
        - "What's the live inventory for Q3?"
        - "List top 5 datacenters by supply"

        I'll find the relevant tables, generate SQL, and execute it for you!"""
    
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    chatbot = cl.user_session.get("chatbot")
    chat_history = cl.user_session.get("chat_history")
    
    # Add user message to chat history
    chat_history.add_user_message(message.content)
    
    # Create and send a loading indicator
    msg = cl.Message(content="")
    await msg.send()
    
    # Process the query
    try:
        result = await chatbot.process_query(message.content, chat_history)
        
        # Prepare response
        response_parts = []
        
        # Step 1: Table selection results
        response_parts.append(f"**📊 Relevant Tables:**\n")
        if result["relevant_tables"]:
            for table in result["relevant_tables"]:
                response_parts.append(f"- `{table}`")
        else:
            response_parts.append("No relevant tables found.")
        
        # Step 2: SQL query
        if result["sql"]:
            response_parts.append(f"\n**🔧 Generated SQL:**\n```sql\n{result['sql']}\n```")
        
        # Step 3: Results
        if result.get("results"):
            if len(result["results"]) > 0 and "error" in result["results"][0]:
                response_parts.append(f"\n**❌ Error:**\n{result['results'][0]['error']}")
            else:
                response_parts.append(f"\n**📈 Results ({len(result['results'])} rows):**")
                
                # Display results in a table format
                if result["results"]:
                    # Create a simple table display
                    table_rows = []
                    headers = list(result["results"][0].keys())
                    
                    # Add headers
                    header_row = "| " + " | ".join(headers) + " |"
                    separator = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"
                    table_rows.append(header_row)
                    table_rows.append(separator)
                    
                    # Add data rows (limit to first 10 for display)
                    for row in result["results"][:10]:
                        values = [str(row.get(h, "")) for h in headers]
                        table_rows.append("| " + " | ".join(values) + " |")
                    
                    if len(result["results"]) > 10:
                        table_rows.append(f"\n*Showing 10 of {len(result['results'])} rows*")
                    
                    response_parts.append("\n".join(table_rows))
        
        # Add assistant response to chat history
        chat_history.add_assistant_message("\n".join(response_parts))
        
        # Update the message with the full response
        await msg.update()
        msg.content = "\n".join(response_parts)
        await msg.update()
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        chat_history.add_assistant_message(error_msg)
        msg.content = error_msg
        await msg.update()

@cl.on_chat_resume
async def on_chat_resume(thread):
    """Handle chat resume."""
    chat_history = ChatHistory()
    # chat_history.add_system_message(SYSTEM_PROMPT)

    # Restore chat history from thread
    # for message in thread["steps"]:
    #     if message["type"] == "user_message":
    #         chat_history.add_user_message(message["output"])
    #     elif message["type"] == "assistant_message":
    #         chat_history.add_assistant_message(message["output"])
    
    # cl.user_session.set("chat_history", chat_history)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

# if __name__ == "__main__":
#     # Run the Chainlit app
#     import chainlit.cli
    
#     # This would typically be run via: chainlit run app.py
#     # For direct execution:
#     from chainlit.cli import run_chainlit
#     run_chainlit(__file__)