# AI Scheduling Assistant

![AI Scheduling Assistant Demo](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWtleXd4bGo0aGFtc2VwMmV1cGJ2cGVmcjlxeGRzNmJ6dHVkcXhzeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2qHveX89Y6ANt9cImN/giphy.gif)
## Overview

This repository contains our submission for the **Agentic AI Scheduling Assistant Hackathon** organized by AMD at IIT Bombay. The challenge was to build an AI-powered assistant capable of autonomously managing scheduling tasks — including understanding user requests, coordinating with calendars, and proposing optimal meeting times with minimal human input.

## Why Agentic AI?

Traditional scheduling tools require a lot of back-and-forth or predefined rules. Our solution uses Agentic AI principles to build an assistant that:

- Understands natural language input (e.g., "Schedule a meeting next Thursday at 2 PM")
- Resolves calendar conflicts and adjusts for time zones
- Learns preferences over time
- Acts autonomously to handle changes, send reminders, and follow up

## Features

- Autonomous meeting coordination with minimal user input  
- Integration with Google Calendar API  
- Natural language processing via LLMs (DeepSeek 7B, LLaMA 3.1)  
- vLLM for high-performance inference on MI300 GPUs  
- JSON-based input/output for seamless system integration  

## Setup Instructions

### Clone and prepare the environment

```bash
git clone https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant.git
cp -r AI-Scheduling-Assistant/* ./
```

### MI300 GPU Setup

Follow the MI300 setup instructions as shown in the provided image:

![MI300 GPU Setup](https://github.com/user-attachments/assets/3b9d68c7-f994-486b-8734-ff61648bb192)

### Calendar Event Extraction

Google Calendar events are extracted using authentication tokens provided during the hackathon. The notebook includes:

- Authentication flow  
- API requests to fetch events  
- Data transformation into clean JSON structure  

Refer to the notebook: [`Calendar_Event_Extraction.ipynb`](https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant/blob/main/Calendar_Event_Extraction.ipynb)

## Running the vLLM Server

### DeepSeek LLM 7B Chat

```bash
HIP_VISIBLE_DEVICES=0 vllm serve /home/user/Models/deepseek-ai/deepseek-llm-7b-chat \
  --gpu-memory-utilization 0.9 \
  --swap-space 16 \
  --disable-log-requests \
  --dtype float16 \
  --max-model-len 2048 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 3000
```

### Meta LLaMA 3.1 8B Instruct

```bash
HIP_VISIBLE_DEVICES=0 vllm serve /home/user/Models/meta-llama/Meta-Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.3 \
  --swap-space 16 \
  --disable-log-requests \
  --dtype float16 \
  --max-model-len 2048 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 4000
```

See more in:  
- [`vLLM_Inference_Servering_DeepSeek.ipynb`](https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant/blob/main/vLLM_Inference_Servering_DeepSeek.ipynb)  
- [`vLLM_Inference_Servering_LLaMA.ipynb`](https://gitenterprise.xilinx.com/asirra/AI-Scheduling-Assistant/blob/main/vLLM_Inference_Servering_LLaMA.ipynb)

## AI Agent Design

Our AI agent is designed to parse natural language emails and extract structured scheduling information. It uses LLM prompts with clear instructions to extract:

- Participants' emails  
- Meeting duration  
- Time constraints  

### Sample Agent Snippet

```python
class AI_AGENT:
    def __init__(self, client, MODEL_PATH):
        self.base_url = BASE_URL
        self.model_path = MODEL_PATH

    def parse_email(self, email_text):
        response = client.chat.completions.create(
            model=self.model_path,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": f"""
                You are an AI scheduling agent. Extract:
                - Email IDs of participants
                - Meeting duration in minutes
                - Time constraints (e.g., next week)

                Format the output strictly as JSON.
                Email: {email_text}
                """
            }]
        )
        return json.loads(response.choices[0].message.content)
```

Notebook: [`Sample_AI_Agent.ipynb`](https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant/blob/main/Sample_AI_Agent.ipynb)

## Input/Output Format

### Input JSON

```json
{
  "Request_id": "6118b54f-907b-4451-8d48-dd13d76033a5",
  "Datetime": "09-07-2025T12:34:55",
  "Location": "IIT Mumbai",
  "From": "userone.amd@gmail.com",
  "Attendees": [
    { "email": "usertwo.amd@gmail.com" },
    { "email": "userthree.amd@gmail.com" }
  ],
  "Subject": "Agentic AI Project Status Update",
  "EmailContent": "Hi team, let's meet on Thursday for 30 minutes to discuss the status of Agentic AI Project."
}
```

### Output JSON

```json
{
  "EventStart": "2025-07-17T10:30:00+05:30",
  "EventEnd": "2025-07-17T11:00:00+05:30",
  "Duration_mins": "30",
  "Attendees": [...],
  "Subject": "...",
  "MetaData": {}
}
```

## Submission Guide

Final submission should be made by executing the following function in the submission notebook:

```python
def your_meeting_assistant(input_json):
    # your implementation
    return {
        "processed": True,
        "output": {...}
    }
```

Notebook: [`Submission.ipynb`](https://github.com/AMD-AI-HACKATHON/AI-Scheduling-Assistant/blob/main/Submission.ipynb)

## Evaluation Criteria

- Accuracy of final output JSON  
- Response time (latency)  
- Code readability and repo structure  
- Creativity in design and approach  
- End-to-end working system  

---

**Note**: This project is developed as part of the AMD AI Sprint Hackathon 2025. The assistant runs on MI300 GPU-backed infrastructure and demonstrates the power of agentic AI in real-world scheduling workflows.
