import openai
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from dataclasses import dataclass
from pydantic import BaseModel
import base64
from pathlib import Path
import asyncio
import logging
import tempfile
from typing import Optional
from dotenv import load_dotenv
import os

# Load environment variables and initialize client
load_dotenv()
#client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
Apikey="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"
client = openai.OpenAI(api_key=Apikey)

# Router setup
router = APIRouter()

# Data models
@dataclass
class ProcessedResponse:
    text: str
    tokens_used: int
    cost: float

class AnalysisResponse(BaseModel):
    text: str
    error: Optional[str] = None
    processing_time: float

# Constants
SYSTEM_PROMPT = """You are the Bevis, and you will be the professional tutor who will help the people to solve the problem.
keep responses BRIEF (< 30 words) and conversational.
focus on key insights only.if you can't clearly see or understand something, say so immediately.
"""

def _calculate_cost(tokens: int) -> float:
    return tokens * 0.01  # adjust rate as needed

async def analyze_image(image_path: Path) -> ProcessedResponse:
    with open(image_path, "rb") as img:
        b64_image = base64.b64encode(img.read()).decode('utf-8')
    try:
        print("sending request to openai")
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "explain this briefly"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100  # keep it tight
        )
        
        text = response.choices[0].message.content
        return ProcessedResponse(
            text=text,
            tokens_used=response.usage.total_tokens,
            cost=_calculate_cost(response.usage.total_tokens)
        )
        
    except openai.APIError as e:
        error_msg = (
            f"OpenAI API error: {str(e)}\n"
            f"Error type: {type(e).__name__}\n"
            f"Status code: {getattr(e, 'status_code', 'N/A')}\n"
            f"Request ID: {getattr(e, 'request_id', 'N/A')}\n"
            f"Headers: {getattr(e, 'headers', {})}"
        )
        logging.error(error_msg)
        raise

    except (openai.APIConnectionError, IOError, OSError, Exception) as e:
        error_msg = f"Error: {str(e)}"
        logging.error(error_msg)
        raise

@router.post("/analyze")
async def analyze_chart(file: UploadFile = File(...)):
    start_time = asyncio.get_event_loop().time()
    
    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)
    
    try:
        result = await analyze_image(temp_path)
        
        return AnalysisResponse(
            text=result.text,
            processing_time=asyncio.get_event_loop().time() - start_time
        )
    
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        temp_path.unlink()  # cleanup temporary file