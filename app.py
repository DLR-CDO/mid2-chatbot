from dotenv import load_dotenv
import os 
import chainlit as cl
from typing import List, Dict, Any, Optional
from semantic_kernel.contents.chat_history import ChatHistory
from nl2sql import NL2SQLChatbot
import pandas as pd
import base64
import httpx
from datetime import datetime
import logging

# ============================================================
# CONFIGURE BASIC LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chainlit_app.log')
    ]
)

logger = logging.getLogger(__name__)

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

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="List providers",
            message="List of providers in Hyderabad?",
            icon="./public/access-onboarding.png",
        ),
        cl.Starter(
            label="Power market share",
            message="Total power market share of NTT in India compared to top 5 providers?",
            icon="./public/workflow_icon.svg",
        ),
        cl.Starter(
            label="Space market share",
            message="Total space market share of DLR in India  compared to top 5 providers?",
            icon="./public/secret-management.png",
        ),
        cl.Starter(
            label="Provider details",
            message="who is the provider for datacenter - Nanxiang Valley Cluster G",
            icon="./public/roles-permissions.png",
        ),
        cl.Starter(
            label="Deployed QoQ",
            message="Total Deployed QoQ region-wise?",
            icon="./public/backup-restore.png",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize chatbot when chat starts."""
    logger.info("Chat started")
    chatbot = NL2SQLChatbot()
    print('user instance :',chatbot)
    cl.user_session.set("chatbot", chatbot)
    
    # Initialize chat history
    chat_history = ChatHistory()
    cl.user_session.set("chat_history", chat_history)
    logger.debug("Chatbot and chat history initialized")


async def send_suggestion_cards(suggestions: List[str]):
    """Send suggestions as clickable cards using CustomElement."""
    if not suggestions:
        return
    
    # Limit to 3 suggestions
    top_suggestions = suggestions[:3]
    
    # Create the custom element with suggestions as props
    suggestion_cards = cl.CustomElement(
        name="SuggestionCards",  # This must match your JSX file name
        props={"suggestions": top_suggestions}
    )
    
    # Send the cards in a message
    await cl.Message(
        content="💡 **Quick follow-up questions:**",
        elements=[suggestion_cards]
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    logger.info(f"Received message: {message.content[:100]}...")
    
    chatbot = cl.user_session.get("chatbot")
    chat_history = cl.user_session.get("chat_history")

    if chatbot is None:
        logger.warning("Chatbot not found in session, creating new one")
        chatbot = NL2SQLChatbot()
        cl.user_session.set("chatbot", chatbot)

    if chat_history is None:
        logger.warning("Chat history not found in session, creating new one")
        chat_history = ChatHistory()
        cl.user_session.set("chat_history", chat_history)
    
    # Add user message to chat history
    chat_history.add_user_message(message.content)
    
    # Create and send a loading indicator
    msg = cl.Message(content="⏳ Processing your query...")
    await msg.send()
    logger.debug("Started processing query")
    
    # Process the query
    try:
        result = await chatbot.process_query(message.content, chat_history)
        cl.user_session.set("last_results", result.get("results"))
        logger.info(f"Query processed. Found {len(result.get('results', []))} results")

        # Prepare response
        response_parts = []
        
        # Step 0: Filter Information
        if result.get("filter_info"):
            filter_display = chatbot.format_filter_info(result["filter_info"])
            if filter_display:
                response_parts.append(f"**🔍 FILTER ANALYSIS**\n{filter_display}")
                logger.debug(f"Filter info: {filter_display}")

        # Step 1: Table selection results
        response_parts.append(f"**📊 Relevant Tables:**\n")
        if result["relevant_tables"]:
            for table in result["relevant_tables"]:
                response_parts.append(f"- `{table}`")
            logger.debug(f"Relevant tables: {result['relevant_tables']}")
        else:
            response_parts.append("No relevant tables found.")
            logger.warning("No relevant tables found for query")
        
        # Step 2: SQL query
        if result["sql"]:
            response_parts.append(f"\n**🔧 Generated SQL:**\n```sql\n{result['sql']}\n```")
            logger.debug(f"Generated SQL: {result['sql'][:200]}...")
        
        # Display results
        if result.get("results"):
            if len(result["results"]) > 0 and "error" in result["results"][0]:
                error_msg = result['results'][0]['error']
                response_parts.append(f"\n**❌ Error:**\n{error_msg}")
                logger.error(f"SQL execution error: {error_msg}")
            else:
                response_parts.append(f"\n**📈 Results ({len(result['results'])} rows):**")
                logger.info(f"Query returned {len(result['results'])} rows")
                
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
        full_response = "\n".join(response_parts)
        chat_history.add_assistant_message(full_response)
        
        # Update the message with the full response
        await msg.update()
        msg.content = full_response
        await msg.update()
        logger.debug("Response sent to user")
        
        # Show suggestions
        if result.get("suggestions") and len(result["suggestions"]) > 0:
            try:
                logger.debug(f"Generated suggestions: {result['suggestions']}")
                await send_suggestion_cards(result["suggestions"])
            except Exception as e:
                logger.error(f"Error sending suggestion cards: {e}", exc_info=True)
                # Fallback: display as plain text
                fallback_msg = "**💡 Suggested questions:**\n\n"
                for i, suggestion in enumerate(result["suggestions"][:3], 1):
                    fallback_msg += f"{i}. {suggestion}\n"
                await cl.Message(content=fallback_msg).send()

        # Show visualization option (after suggestions)
        if result.get("results") and len(result["results"]) > 0:
            if len(result["results"]) > 0 and "error" not in result["results"][0]:
                logger.debug("Showing visualization option")
                await cl.Message(
                    content="📊 Want to visualize this data?",
                    actions=[
                        cl.Action(
                            name="visualize",
                            payload={},
                            label="Visualize"
                        )
                    ]
                ).send()
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(f"Error processing message: {e}", exc_info=True)
        chat_history.add_assistant_message(error_msg)
        msg.content = error_msg
        await msg.update()

# Replace the entire on_visualize function in app.py with this:

@cl.action_callback("visualize")
async def on_visualize(action: cl.Action):
    """Handle visualization request with Plotly."""
    logger.info("Visualization action triggered")
    
    chatbot = cl.user_session.get("chatbot")
    results = cl.user_session.get("last_results")
    chat_history = cl.user_session.get("chat_history")
    
    if not results:
        logger.warning("No results available for visualization")
        await cl.Message("No data available to visualize.").send()
        return
    
    # Get the last user query
    last_user_message = None
    for msg in reversed(chat_history.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break
    
    if not last_user_message:
        last_user_message = "Visualize this data"
    
    logger.info(f"Generating Plotly visualization for: {last_user_message[:50]}...")
    
    # Show processing message
    msg = cl.Message(content="📊 Generating interactive visualization...")
    await msg.send()
    
    # Generate chart data with Plotly figure
    chart_data = await chatbot.prepare_chart_data(results, last_user_message)
    
    # Handle error
    if "error" in chart_data:
        error_msg = chart_data["error"]
        logger.warning(f"Visualization failed: {error_msg}")
        
        await msg.update()
        msg.content = f"**⚠️ Visualization Error**\n\n{error_msg}"
        await msg.update()
        return
    
    # Check for Plotly type
    if chart_data.get("type") == "plotly":
        try:
            # Create Plotly element
            plotly_element = cl.Plotly(
                name="chart",
                figure=chart_data["figure"],
                display="inline"
            )
            
            await msg.update()
            msg.content = "📊 **Chart Generated**"
            await msg.update()
            
            # Send the Plotly chart
            await cl.Message(
                content="",
                elements=[plotly_element]
            ).send()
            
            # Send chart info
            info_parts = [
                f"**Chart Details**",
                f"• **Data Points:** {chart_data.get('rows', 0):,} rows",
                f"• **Columns:** {', '.join(chart_data.get('columns', []))}"
            ]
            
            # Add interaction tips
            info_parts.extend([
                "",
                "**💡 Interaction Tips:**",
                "• **Hover** over points to see details",
                "• **Zoom** by dragging a rectangle",
                "• **Pan** by dragging the chart",
                "• **Reset** by double-clicking",
                "• **Download** as PNG via the camera icon"
            ])
            
            await cl.Message(content="\n".join(info_parts)).send()
            
        except Exception as e:
            logger.error(f"Error displaying Plotly chart: {e}", exc_info=True)
            await msg.update()
            msg.content = f"Failed to display chart: {str(e)}"
            await msg.update()
    else:
        await msg.update()
        msg.content = "Unknown chart type returned."
        await msg.update()

# @cl.action_callback("visualize")
# async def on_visualize(action: cl.Action):
#     """Handle visualization request with HTML chart."""
#     logger.info("Visualization action triggered")
    
#     chatbot = cl.user_session.get("chatbot")
#     results = cl.user_session.get("last_results")
#     chat_history = cl.user_session.get("chat_history")
    
#     if not results:
#         logger.warning("No results available for visualization")
#         await cl.Message("No data available to visualize.").send()
#         return
    
#     # Get the last user query
#     last_user_message = None
#     for msg in reversed(chat_history.messages):
#         if msg.role == "user":
#             last_user_message = msg.content
#             break
    
#     if not last_user_message:
#         last_user_message = "Visualize this data"
    
#     logger.info(f"Generating HTML visualization for: {last_user_message[:50]}...")
    
#     # Show processing message
#     msg = cl.Message(content="📊 Generating interactive visualization...")
#     await msg.send()
    
#     # Generate chart and save as HTML
#     chart_data = await chatbot.prepare_chart_data(results, last_user_message)
    
#     # Handle error
#     if "error" in chart_data:
#         error_msg = chart_data["error"]
#         logger.warning(f"Visualization failed: {error_msg}")
        
#         await msg.update()
#         msg.content = f"**⚠️ Visualization Error**\n\n{error_msg}"
#         await msg.update()
#         return
    
#     # Check for HTML type
#     if chart_data.get("type") == "html":
#         html_path = chart_data.get("local_path")
#         if html_path and os.path.exists(html_path):
#             # Read the HTML content
#             with open(html_path, "r", encoding="utf-8") as f:
#                 html_content = f.read()
            
#             await msg.update()
#             msg.content = "📊 **Interactive Visualization**"
#             await msg.update()
            
#             # Method 1: Direct HTML embedding using cl.Html
#             try:
#                 # Create an HTML element with the chart
#                 html_element = cl.Html(
#                     content=html_content,
#                     name="chart",
#                     display="inline"
#                 )
                
#                 # Send the HTML element
#                 await cl.Message(
#                     content="",
#                     elements=[html_element]
#                 ).send()
                
#             except Exception as e:
#                 logger.error(f"Error with cl.Html: {e}")
#                 # Fallback: Use iframe
#                 try:
#                     # Try using Element with type="html"
#                     html_element = cl.Element(
#                         name="chart_iframe",
#                         type="html",
#                         content=html_content,
#                         display="inline"
#                     )
                    
#                     await cl.Message(
#                         content="",
#                         elements=[html_element]
#                     ).send()
                    
#                 except Exception as e2:
#                     logger.error(f"Error with cl.Element: {e2}")
#                     # Last resort: provide download link
#                     html_relative_path = chart_data.get("html_path", "")
#                     chart_url = f"/public/{html_relative_path}"
                    
#                     await cl.Message(
#                         content=f"📊 **Chart Generated**\n\nYou can view the interactive chart here: [{chart_url}]({chart_url})"
#                     ).send()
            
#             # Send chart info
#             info_parts = [
#                 f"**Chart Type:** Interactive HTML Chart",
#                 f"**Data Points:** {chart_data.get('rows', 0):,} rows",
#                 f"**Columns:** {', '.join(chart_data.get('columns', [])[:5])}"
#             ]
            
#             if len(chart_data.get('columns', [])) > 5:
#                 info_parts[-1] += f" ... (+{len(chart_data['columns']) - 5} more)"
            
#             await cl.Message(content="\n".join(info_parts)).send()
#         else:
#             await msg.update()
#             msg.content = "Chart HTML file not found."
#             await msg.update()
#     else:
#         await msg.update()
#         msg.content = "Unknown chart type returned."
#         await msg.update()
 