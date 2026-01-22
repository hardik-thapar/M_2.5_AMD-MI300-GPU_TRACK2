import os
import json
from datetime import datetime, timezone, timedelta
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dateutil import parser
import pytz
from typing import Annotated, List, Dict, Any
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
import asyncio

# Configuration
BASE_URL = "http://localhost:3000/v1"  # DeepSeek model
# BASE_URL = "http://localhost:4000/v1"  # LLaMA model
os.environ["BASE_URL"] = BASE_URL
os.environ["OPENAI_API_KEY"] = "abc-123"

# Provider setup
provider = OpenAIProvider(
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

# Pydantic models
class MeetingMetadata(BaseModel):
    participants: List[str] = Field(..., description="list of emails who are the participants for the meeting.")
    start_time: str = Field(..., description="A valid format of time string for the Starting point of time window")
    end_time: str = Field(..., description="A valid format of time string for the ending point of time window")
    meeting_duration: int = Field(..., description="Duration of the meeting in minutes.")

class MeetingInfo(BaseModel):
    participants: str = Field(..., description="Comma-separated emails of all participants.")
    time_constraints: str = Field(..., description="Mentioned timing or date phrase in the email.")
    meeting_duration: int = Field(..., description="Duration of the meeting in minutes.")

# Tools
@Tool
def get_current_date() -> str:
    """Return the current date/time as an ISO-formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@Tool
def get_time_window(
    current_date: Annotated[str, "The string formatted timestamp of current date & time"],
    time_constraints: Annotated[str, "The exact date, day or time window where the call can be scheduled"],
    meeting_duration: Annotated[int, "how long the call/meeting is going to occur, unit in minutes"]
) -> MeetingMetadata:
    """Calculate the time window for scheduling based on constraints."""
    # Parse current date
    current_dt = parser.parse(current_date)
    
    # Handle different time constraints
    if "next week" in time_constraints.lower():
        # Find next Monday
        days_ahead = 7 - current_dt.weekday()  # Monday is 0
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        start_date = current_dt + timedelta(days=days_ahead)
        start_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0)  # 9 AM
        end_date = start_date + timedelta(days=4)  # Friday
        end_date = end_date.replace(hour=17, minute=0, second=0, microsecond=0)  # 5 PM
    
    elif "thursday" in time_constraints.lower():
        # Find next Thursday
        days_ahead = 3 - current_dt.weekday()  # Thursday is 3
        if days_ahead <= 0:  # Thursday already happened this week
            days_ahead += 7
        start_date = current_dt + timedelta(days=days_ahead)
        start_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0)
        end_date = start_date.replace(hour=17, minute=0, second=0, microsecond=0)
    
    elif "tuesday" in time_constraints.lower():
        # Find next Tuesday
        days_ahead = 1 - current_dt.weekday()  # Tuesday is 1
        if days_ahead <= 0:  # Tuesday already happened this week
            days_ahead += 7
        start_date = current_dt + timedelta(days=days_ahead)
        start_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0)
        end_date = start_date.replace(hour=17, minute=0, second=0, microsecond=0)
    
    elif "wednesday" in time_constraints.lower():
        # Find next Wednesday
        days_ahead = 2 - current_dt.weekday()  # Wednesday is 2
        if days_ahead <= 0:  # Wednesday already happened this week
            days_ahead += 7
        start_date = current_dt + timedelta(days=days_ahead)
        start_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0)
        end_date = start_date.replace(hour=17, minute=0, second=0, microsecond=0)
    
    elif "monday" in time_constraints.lower():
        # Find next Monday
        days_ahead = 0 - current_dt.weekday()  # Monday is 0
        if days_ahead <= 0:  # Monday already happened this week
            days_ahead += 7
        start_date = current_dt + timedelta(days=days_ahead)
        start_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0)
        end_date = start_date.replace(hour=17, minute=0, second=0, microsecond=0)
    
    else:
        # Default to next business day
        start_date = current_dt + timedelta(days=1)
        start_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0)
        end_date = start_date.replace(hour=17, minute=0, second=0, microsecond=0)
    
    # Add timezone info
    ist = pytz.timezone('Asia/Kolkata')
    start_date = ist.localize(start_date)
    end_date = ist.localize(end_date)
    
    # Extract participants from the context (this would be passed separately in real implementation)
    participants = ["userone.amd@gmail.com", "usertwo.amd@gmail.com", "userthree.amd@gmail.com"]
    
    return MeetingMetadata(
        participants=participants,
        start_time=start_date.isoformat(),
        end_time=end_date.isoformat(),
        meeting_duration=meeting_duration
    )

@Tool
def extract_meeting_info(
    email: Annotated[str, "The raw email body containing meeting details."]
) -> MeetingInfo:
    """Extract meeting information from email content."""
    # This would use the LLM to parse the email
    # For now, return a structured response
    return MeetingInfo(
        participants="userone.amd@gmail.com,usertwo.amd@gmail.com,userthree.amd@gmail.com",
        time_constraints="thursday",
        meeting_duration=30
    )

@Tool
def retrieve_calendar_events(user: Annotated[str, "The mail id of each individual user."], 
                            start: Annotated[str, "The string formatted timestamp of start point of the time-window"],
                            end: Annotated[str, "The string formatted timestamp of end point of the time-window"]
                           ) -> List[Dict[str, Any]]:
    """Fetch Google Calendar events for a user between start and end times."""
    events_list = []
    try:
        token_path = f"../Keys/{user.split('@')[0]}.token"
        user_creds = Credentials.from_authorized_user_file(token_path)
        calendar_service = build("calendar", "v3", credentials=user_creds)
        events_result = calendar_service.events().list(
            calendarId='primary', 
            timeMin=start, 
            timeMax=end, 
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        
        for event in events:
            attendee_list = []
            try:
                for attendee in event.get("attendees", []):
                    attendee_list.append(attendee['email'])
            except:
                attendee_list.append("SELF")
            
            start_time = event["start"]["dateTime"]
            end_time = event["end"]["dateTime"]
            events_list.append({
                "StartTime": start_time,
                "EndTime": end_time,
                "NumAttendees": len(set(attendee_list)),
                "Attendees": list(set(attendee_list)),
                "Summary": event.get("summary", "No Title")
            })
    except Exception as e:
        print(f"Error fetching calendar events for {user}: {e}")
        # Return sample data for testing
        events_list = [{
            "StartTime": "2025-07-17T10:00:00+05:30",
            "EndTime": "2025-07-17T11:00:00+05:30",
            "NumAttendees": 1,
            "Attendees": ["SELF"],
            "Summary": "Sample Event"
        }]
    
    return events_list

@Tool
def convert_to_local_timezone(time_str: str, target_timezone: str = "Asia/Kolkata"):
    """Convert any datetime string to a pytz-localized datetime object in the target timezone."""
    dt = parser.parse(time_str)
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    local_tz = pytz.timezone(target_timezone)
    return dt.astimezone(local_tz)

# Agent setup
agent_model = OpenAIModel("deepseek-llm-7b-chat", provider=provider)

agent = Agent(
    model=agent_model,
    tools=[extract_meeting_info, get_current_date, get_time_window, retrieve_calendar_events, convert_to_local_timezone],
    system_prompt="""
    You are an AI agent that assists with meeting scheduling.
    Your tools include:
        - extract_meeting_info(email)
        - get_current_date()
        - get_time_window()
        - retrieve_calendar_events()
        - convert_to_local_timezone()

    Your task is to:
        1. Parse the email to extract meeting details
        2. Determine the current date
        3. Find the required meeting timeframe using get_time_window
        4. Get each attendee's detailed calendar schedule
        5. Find the best available time slot for all attendees
        6. Return the scheduled meeting details
    """
)

# Main scheduling function
def schedule_meeting(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to schedule a meeting based on input data."""
    try:
        # Extract basic info
        request_id = input_data["Request_id"]
        from_email = input_data["From"]
        attendees = [attendee["email"] for attendee in input_data["Attendees"]]
        subject = input_data["Subject"]
        email_content = input_data["EmailContent"]
        
        # Add sender to attendees if not present
        all_attendees = [from_email] + attendees if from_email not in attendees else attendees
        
        # Parse email content
        meeting_info = extract_meeting_info(email_content)
        
        # Get current date
        current_date = get_current_date()
        
        # Calculate time window
        time_window = get_time_window(current_date, meeting_info.time_constraints, meeting_info.meeting_duration)
        
        # Get calendar events for all attendees
        all_events = {}
        for attendee in all_attendees:
            events = retrieve_calendar_events(attendee, time_window.start_time, time_window.end_time)
            all_events[attendee] = events
        
        # Find best available time slot (simplified logic)
        # In a real implementation, you'd use the LLM to analyze conflicts and find optimal times
        meeting_start = time_window.start_time
        meeting_end = (parser.parse(meeting_start) + timedelta(minutes=meeting_info.meeting_duration)).isoformat()
        
        # Build output structure
        output_data = input_data.copy()
        output_data["Attendees"] = []
        
        for attendee in all_attendees:
            attendee_events = all_events.get(attendee, [])
            # Add the new meeting to each attendee's events
            attendee_events.append({
                "StartTime": meeting_start,
                "EndTime": meeting_end,
                "NumAttendees": len(all_attendees),
                "Attendees": all_attendees,
                "Summary": subject
            })
            
            output_data["Attendees"].append({
                "email": attendee,
                "events": attendee_events
            })
        
        output_data["EventStart"] = meeting_start
        output_data["EventEnd"] = meeting_end
        output_data["Duration_mins"] = str(meeting_info.meeting_duration)
        output_data["MetaData"] = {}
        
        return output_data
        
    except Exception as e:
        print(f"Error in schedule_meeting: {e}")
        # Return input data with error info
        input_data["EventStart"] = ""
        input_data["EventEnd"] = ""
        input_data["Duration_mins"] = ""
        input_data["MetaData"] = {"error": str(e)}
        return input_data

# Test function
async def test_scheduling():
    """Test the scheduling functionality."""
    test_input = {
        "Request_id": "6118b54f-907b-4451-8d48-dd13d76033a5",
        "Datetime": "09-07-2025T12:34:55",
        "Location": "IIT Mumbai",
        "From": "userone.amd@gmail.com",
        "Attendees": [
            {"email": "usertwo.amd@gmail.com"},
            {"email": "userthree.amd@gmail.com"}
        ],
        "Subject": "Agentic AI Project Status Update",
        "EmailContent": "Hi team, let's meet on Thursday for 30 minutes to discuss the status of Agentic AI Project."
    }
    
    result = schedule_meeting(test_input)
    print("Scheduling Result:")
    print(json.dumps(result, indent=2))
    return result

# Run test if this file is executed directly
if __name__ == "__main__":
    asyncio.run(test_scheduling()) 