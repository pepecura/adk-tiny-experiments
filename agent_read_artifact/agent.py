# Tiny experiment for uploading a file on ADK web to root agent as an artifact and passing that file to another agent for processing.
# Key components for the task: SaveFilesAsArtifactsPlugin(), load_artifacts
# Heads up: App name will be the same as the agent folder name (highlighted in the comments).

import os,sys,io
from google.adk.agents import Agent
from google.adk.tools import ToolContext, AgentTool, load_artifacts
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google import genai
from google.genai import types

async def extract_title_with_gemini(tool_context: ToolContext) -> str:
    """
    Sends PDF bytes to Gemini 2.5 Flash to extract the title intelligently.
    Arguments:
        tool_context: The ADK context object used to load the artifact.
    Returns a string containing the title.
    """
    artifacts = await tool_context.list_artifacts()
    print("FILENAMES: ", artifacts)
    most_recent_file = artifacts[-1]

    try:

        # Get the file from ADK Artifact Service
        artifact_content = await tool_context.load_artifact(filename=most_recent_file)
        file_name = artifact_content.inline_data.display_name
        data_bytes = artifact_content.inline_data.data

        client = genai.Client(
            vertexai=True, 
            project='XXX', 
            location='XXX'
        )
        
        # Prepare the Prompt
        prompt = "Extract the main title of this document. Return ONLY the title text, nothing else."

        # Call Gemini
        # Send the PDF bytes + a text instruction as one request.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                # Convert raw bytes to a GenAI Part
                types.Part.from_bytes(
                    data=data_bytes, 
                    mime_type="application/pdf"
                ),
                prompt
            ]
        )

        # Return the result
        return response.text.strip()

    except Exception as e:
        return f"Gemini extraction failed: {e}"


# --- Specialist Agent: Title Extractor Agent ---
title_extractor_agent = Agent(
    name="title_extractor",
    model="gemini-2.5-flash", # Ensure this model matches what is enabled in your Vertex project
    instruction="""
    You are a Metadata Specialist.
    1. Use 'extract_title_with_gemini' to read the file and extract the title.
    2. Return ONLY the extracted title.
    """,
    tools=[extract_title_with_gemini]
)

# --- Root Agent ---
root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    description=("Orchestrator"),
    instruction=('''
    You will help with the uploading of a file and reading the title in the file.
    Upon a user request
    * ask the user to upload the file
    * wait for load_artifacts to complete
    * call title_extractor_agent with the tool context
    * Respond to the user with the returned string from title_extractor_agent
    '''
    ),
    tools=[load_artifacts, AgentTool(title_extractor_agent)],
)

app = App(
    name="agent_read_artifact", # --> SHOULD BE the same name as the agent folder name, otherwise you get session not found error.
    root_agent=root_agent,
    plugins=[SaveFilesAsArtifactsPlugin()],
)
