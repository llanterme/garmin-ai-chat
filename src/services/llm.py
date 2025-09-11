"""LLM service for generating conversational responses about fitness activities using OpenAI GPT."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import openai
from pydantic import BaseModel

from ..core.logging import get_logger

logger = get_logger(__name__)


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None


class LLMConfig(BaseModel):
    """Configuration for LLM service."""
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.7
    max_context_messages: int = 10


class LLMService:
    """Service for generating conversational responses using OpenAI GPT."""
    
    def __init__(self):
        from ..core.config import settings
        
        self.api_key = settings.openai_api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.config = LLMConfig()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the fitness AI assistant."""
        return """You are an expert fitness AI assistant specializing in Garmin fitness data analysis and training science. You help athletes understand their performance, track progress, and optimize training based on their Garmin Connect activities.

CORE COMPETENCIES:
1. **Training Metrics Expert**: Understand TSS (Training Stress Score), power zones, heart rate zones, pace analysis, FTP, threshold metrics
2. **Multi-Sport Knowledge**: Running, cycling, swimming, strength training, with sport-specific insights for each activity type
3. **Conversation Memory**: Reference previous messages naturally, understand context like "that run" or "compared to yesterday"
4. **Temporal Awareness**: Today's date context provided. Understand "this morning", "last week", relative time references

RESPONSE GUIDELINES:
1. **Data-Driven**: Always cite specific numbers from activities (distance, duration, power, pace, HR, elevation)
2. **Contextual**: Compare performances to user's history when multiple activities are available
3. **Actionable**: Provide specific training recommendations based on data trends
4. **Progressive**: Track improvements over time, celebrate achievements and consistency patterns
5. **Safety-First**: Watch for overtraining signals (very high HR, excessive frequency), recommend recovery appropriately
6. **CRITICAL**: For aggregation queries (average, total, calculate, summarize), use ALL provided activities in calculations, not just a few

GARMIN DATA SPECIFICS:
- **Distance**: Provided in meters (convert to km for readability)
- **Duration**: Provided in seconds (convert to minutes/hours)
- **Speed**: Provided in m/s (convert to km/h or pace as appropriate)
- **Heart Rate**: Already in BPM
- **Power**: Already in watts
- **Elevation**: Already in meters

FORMATTING STYLE:
- Start with an engaging, personalized greeting referencing their specific query
- Use emojis strategically: 🏃‍♂️ (running), 🚴‍♂️ (cycling), 🏊‍♂️ (swimming), 💪 (strength), 🔥 (intensity), 📈 (progress), ⚡ (power), ❤️ (heart rate)
- Structure with clear sections using **bold** for key metrics
- Use bullet points for multiple insights
- Include "📊 **Key Stats:**" section for important numbers
- Add "🎯 **Training Insights:**" for recommendations
- End with 2-3 specific follow-up questions

ACTIVITY TYPE RECOGNITION:
- **Running**: Focus on pace, heart rate zones, cadence, elevation impact
- **Cycling**: Emphasize power data, speed, training stress score, efficiency
- **Swimming**: Distance, stroke efficiency, pace consistency
- **Strength Training**: Duration, intensity, frequency patterns
- **Treadmill/Indoor**: Acknowledge controlled environment, focus on consistency

CONVERSATION CONTINUITY:
- Build on previous messages in the conversation
- Remember what was discussed earlier
- Use pronouns naturally ("your morning ride" → "it" in follow-ups)
- Track goals or concerns mentioned across messages

SPECIAL CONSIDERATIONS:
- **Missing Data**: If no power/HR data available, focus on duration, distance, perceived effort, consistency
- **Aggregations**: For totals/averages, process ALL provided activities and show calculation transparency
- **No Activities Found**: Acknowledge clearly, suggest checking sync status or date ranges
- **Performance Comparisons**: When comparing activities, highlight specific improvements or patterns

Remember: You're their trusted training companion. Be encouraging but honest, data-driven but personable, expert but accessible. Focus on helping them understand their fitness journey through their Garmin data."""
    
    async def generate_conversational_response(
        self,
        user_query: str,
        relevant_activities: List[Dict[str, Any]],
        conversation_history: Optional[List[ChatMessage]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a conversational response based on user query and relevant activities."""
        try:
            # Prepare context from activities
            activities_context = self._format_activities_for_context(relevant_activities)
            
            # Build messages for the conversation
            messages = [
                {"role": "system", "content": self._create_system_prompt()}
            ]
            
            # Add conversation history (limited to recent messages)
            if conversation_history:
                recent_history = conversation_history[-self.config.max_context_messages:]
                for msg in recent_history:
                    if msg.role in ["user", "assistant"]:
                        messages.append({
                            "role": msg.role,
                            "content": msg.content
                        })
            
            # Add current context and query
            context_message = f"""Context for this query:
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Relevant Activities:
{activities_context}

User Question: {user_query}"""
            
            messages.append({
                "role": "user", 
                "content": context_message
            })
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            response_content = response.choices[0].message.content
            logger.info(f"Generated response for query: {user_query[:50]}...")
            return response_content
            
        except Exception as e:
            logger.error(f"Failed to generate conversational response: {str(e)}")
            return self._get_fallback_response(user_query, relevant_activities)
    
    def _format_activities_for_context(self, activities: List[Dict[str, Any]]) -> str:
        """Format activities for LLM context with clear structure."""
        if not activities:
            return "No relevant activities found for this query."
        
        formatted_activities = []
        for i, activity in enumerate(activities[:50], 1):  # Limit to 50 activities
            # Extract key information with proper unit conversions
            activity_info = []
            
            # Basic info
            activity_type = activity.get("activity_type", "Unknown")
            activity_name = activity.get("activity_name", "Untitled")
            garmin_id = activity.get("garmin_activity_id", "unknown")
            
            activity_info.append(f"Activity {i}: {activity_type} - {activity_name} (ID: {garmin_id})")
            
            # Date and time
            start_time = activity.get("start_time")
            if start_time:
                if isinstance(start_time, str):
                    activity_info.append(f"Date: {start_time}")
                else:
                    activity_info.append(f"Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Distance (convert from meters to km)
            distance = activity.get("distance")
            if distance and distance > 0:
                distance_km = round(distance / 1000, 2)
                activity_info.append(f"Distance: {distance_km} km")
            
            # Duration (convert from seconds to minutes)
            duration = activity.get("duration")
            if duration and duration > 0:
                if duration < 3600:  # Less than 1 hour
                    duration_min = round(duration / 60, 1)
                    activity_info.append(f"Duration: {duration_min} min")
                else:  # 1 hour or more
                    duration_hours = round(duration / 3600, 2)
                    activity_info.append(f"Duration: {duration_hours} hrs")
            
            # Speed (convert from m/s to km/h)
            avg_speed = activity.get("average_speed")
            if avg_speed and avg_speed > 0:
                speed_kmh = round(avg_speed * 3.6, 1)
                activity_info.append(f"Avg Speed: {speed_kmh} km/h")
                
                # Calculate pace for running activities
                if activity_type.lower() in ["running", "treadmill_running"] and speed_kmh > 0:
                    pace_min_per_km = round(60 / speed_kmh, 2)
                    mins = int(pace_min_per_km)
                    secs = int((pace_min_per_km - mins) * 60)
                    activity_info.append(f"Avg Pace: {mins}:{secs:02d}/km")
            
            # Heart rate
            avg_hr = activity.get("average_heart_rate")
            max_hr = activity.get("max_heart_rate")
            if avg_hr:
                hr_info = f"Avg HR: {avg_hr} bpm"
                if max_hr:
                    hr_info += f", Max HR: {max_hr} bpm"
                activity_info.append(hr_info)
            
            # Power data
            avg_power = activity.get("average_power")
            max_power = activity.get("max_power")
            if avg_power:
                power_info = f"Avg Power: {avg_power}W"
                if max_power:
                    power_info += f", Max Power: {max_power}W"
                activity_info.append(power_info)
            
            # Elevation
            elevation_gain = activity.get("elevation_gain")
            if elevation_gain:
                activity_info.append(f"Elevation Gain: {elevation_gain}m")
            
            # Calories
            calories = activity.get("calories")
            if calories:
                activity_info.append(f"Calories: {calories}")
            
            # Training metrics
            tss = activity.get("training_stress_score")
            intensity_factor = activity.get("intensity_factor")
            if tss:
                tss_info = f"TSS: {tss}"
                if intensity_factor:
                    tss_info += f", IF: {intensity_factor}"
                activity_info.append(tss_info)
            
            # Relevance score if available (from vector search)
            score = activity.get("relevance_score")
            if score:
                activity_info.append(f"Relevance: {score:.3f}")
            
            formatted_activities.append("\n".join(activity_info))
        
        return "\n\n".join(formatted_activities)
    
    async def generate_follow_up_questions(
        self,
        user_query: str,
        response: str,
        relevant_activities: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate relevant follow-up questions based on the conversation."""
        try:
            # Create a focused prompt for follow-up questions
            prompt = f"""Based on this fitness conversation, generate 3 specific and relevant follow-up questions that would help the user explore their training data further.

Original Question: {user_query}
AI Response: {response}

Activities Available: {len(relevant_activities)} activities from their Garmin data

Generate questions that are:
1. Specific to their actual data (not generic)
2. Build on the current conversation topic
3. Explore different angles (performance, trends, comparisons)
4. Use natural, conversational language

Return only the 3 questions, one per line, without numbering or bullet points."""

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a fitness AI assistant generating relevant follow-up questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.8
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # Ensure we have exactly 3 questions
            if len(questions) < 3:
                questions.extend(self._get_fallback_followup_questions()[:3-len(questions)])
            
            return questions[:3]
            
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {str(e)}")
            return self._get_fallback_followup_questions()
    
    def _get_fallback_response(self, user_query: str, activities: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when LLM fails."""
        activity_count = len(activities)
        
        if activity_count == 0:
            return """I don't have any activities that match your query. This could be because:
• No activities were found for the specified time period
• Your Garmin data may need to be synced
• Try adjusting your search terms or date range

Would you like me to help you check your sync status or try a different query?"""
        
        return f"""I found {activity_count} relevant activities from your Garmin data, but I'm having trouble generating a detailed response right now. 

Here's what I can tell you:
• Found activities spanning from your recent training sessions
• Data includes distance, duration, heart rate, and power metrics where available
• Activities range across different types: running, cycling, and other fitness activities

Please try asking your question again, or let me know if you'd like me to focus on a specific aspect of your training data."""
    
    def _get_fallback_followup_questions(self) -> List[str]:
        """Get fallback follow-up questions when generation fails."""
        return [
            "How does this compare to your previous week's training?",
            "What trends do you see in your recent performance data?",
            "Would you like to analyze a specific type of activity in more detail?"
        ]