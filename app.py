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
    chatbot = NL2SQLChatbot()
    print('user instance :',chatbot)
    cl.user_session.set("chatbot", chatbot)
    
    # Initialize chat history
    chat_history = ChatHistory()
    cl.user_session.set("chat_history", chat_history)


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
    chatbot = cl.user_session.get("chatbot")
    chat_history = cl.user_session.get("chat_history")

    if chatbot is None:
        chatbot = NL2SQLChatbot()
        cl.user_session.set("chatbot", chatbot)

    if chat_history is None:
        chat_history = ChatHistory()
        cl.user_session.set("chat_history", chat_history)
    
    # Add user message to chat history
    chat_history.add_user_message(message.content)
    
    # Create and send a loading indicator
    msg = cl.Message(content="⏳ Processing your query...")
    await msg.send()
    
    # Process the query
    try:
        result = await chatbot.process_query(message.content, chat_history)
        cl.user_session.set("last_results", result.get("results"))

        # Prepare response
        response_parts = []
        
        # Step 0: Filter Information
        if result.get("filter_info"):
            filter_display = chatbot.format_filter_info(result["filter_info"])
            if filter_display:
                response_parts.append(f"**🔍 FILTER ANALYSIS**\n{filter_display}")

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
        
        # Display results
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
        full_response = "\n".join(response_parts)
        chat_history.add_assistant_message(full_response)
        
        # Update the message with the full response
        await msg.update()
        msg.content = full_response
        await msg.update()
        
        # NEW ORDER: Show suggestions FIRST (before visualization)
        if result.get("suggestions") and len(result["suggestions"]) > 0:
            try:
                await send_suggestion_cards(result["suggestions"])
            except Exception as e:
                print(f"Error sending suggestion cards: {e}")
                # Fallback: display as plain text
                fallback_msg = "**💡 Suggested questions:**\n\n"
                for i, suggestion in enumerate(result["suggestions"][:3], 1):
                    fallback_msg += f"{i}. {suggestion}\n"
                await cl.Message(content=fallback_msg).send()

        # THEN show visualization option (after suggestions)
        if result.get("results") and len(result["results"]) > 0:
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
        chat_history.add_assistant_message(error_msg)
        msg.content = error_msg
        await msg.update()


@cl.action_callback("visualize")
async def on_visualize(action: cl.Action):
    chatbot = cl.user_session.get("chatbot")
    results = cl.user_session.get("last_results")
    chat_history = cl.user_session.get("chat_history")
    
    if not results:
        await cl.Message("No data available to visualize.").send()
        return
    
    # Get the last user query from chat history
    last_user_message = None
    for msg in reversed(chat_history.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break
    
    if not last_user_message:
        last_user_message = "Visualize this data"
    
    # Step 1: Prepare chart data
    chart_data = await chatbot.prepare_chart_data(results, last_user_message)
    
    if "error" in chart_data:
        await cl.Message(f"⚠️ {chart_data['error']}").send()
        return
    
    # Show processing message
    msg = cl.Message(content="🖼️ Generating visualization...")
    await msg.send()
    
    # Step 2: Build image prompt from chart_spec (INLINE)
    chart_spec = chart_data["chart_spec"]
    columns = chart_data["columns"]

    # Minimal safety check
    if (
        chart_spec.get("x_axis") not in columns or
        chart_spec.get("y_axis") not in columns
    ):
        await msg.update()
        msg.content = "⚠️ Cannot visualize this data safely."
        await msg.update()
        return

    image_prompt = f"""
    Create a {chart_spec['chart_type']} chart.

    Title: {chart_spec['title']}
    X-axis: {chart_spec['x_axis']}
    Y-axis: {chart_spec['y_axis']}

    {"Series: " + ", ".join(chart_spec['series']) if chart_spec.get("series") else ""}

    Design:
    - Theme: {chart_spec['design']['theme']}
    - Palette: {chart_spec['design']['palette']}
    - Legend: {chart_spec['design']['show_legend']}

    Rules:
    - Use ONLY the specified axes and series
    - Do NOT invent categories or values
    - Clean professional dashboard style
    """

    image_result = await chatbot.generate_chart_image(image_prompt)

    if "error" in image_result:
        await msg.update()
        msg.content = f"⚠️ Visualization failed: {image_result['error']}"
        await msg.update()
        return
    
    # Step 3: Save and display image
    try:
        # Decode and save image
        public_dir = "public"
        os.makedirs(public_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"chart_{timestamp}.png"
        img_path = os.path.join(public_dir, img_filename)
        
        image_bytes = base64.b64decode(image_result["base64_image"])
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        
        # Display image in Chainlit
        image = cl.Image(
            name=img_filename,
            path=img_path,
            display="inline",
            size="large"
        )
        
        await msg.update()
        msg.content = "📊 **Visualization Generated**"
        await msg.update()
        
        # Send image
        await cl.Message(
            content="",
            elements=[image]
        ).send()
        
        # Send chart info
        await cl.Message(
            f"**Chart Type:** {chart_data['chart_spec']['chart_type']}\n"
            f"**Data Points:** {chart_data['rows']} rows × {len(chart_data['columns'])} columns"
        ).send()
        
    except Exception as e:
        await msg.update()
        msg.content = f"⚠️ Failed to save/display image: {str(e)}"
        await msg.update()

@cl.on_chat_resume
async def on_chat_resume(thread):
    """Handle chat resume."""
    chat_history = ChatHistory()
    cl.user_session.set("chat_history", chat_history)

 