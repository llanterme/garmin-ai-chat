"""
LLM service for generating conversational responses about fitness activities.
"""

import os
from typing import List, Dict, Any, Optional
import openai
from datetime import datetime
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None


class LLMService:
    """Service for generating conversational responses using OpenAI GPT."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"  # Cost-effective model for conversations
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the fitness AI assistant."""
        return """You are an expert fitness AI assistant specializing in endurance sports analytics, with deep knowledge of Strava data and training science. You help athletes understand their performance, track progress, and optimize training.

CORE COMPETENCIES:
1. **Training Metrics Expert**: Understand TSS, CTL (Fitness), ATL (Fatigue), TSB (Form), FTP, threshold power/pace, heart rate zones
2. **Multi-Sport Knowledge**: Cycling (indoor/outdoor), running, swimming, triathlon, with sport-specific insights
3. **Conversation Memory**: Reference previous messages naturally, understand context like "that ride" or "compared to yesterday"
4. **Temporal Awareness**: Today's date context provided. Understand "this morning", "last week", relative time references

RESPONSE GUIDELINES:
1. **Data-Driven**: Always cite specific numbers from activities (distance, duration, TSS, power, pace, HR)
2. **Contextual**: Compare performances to user's history and typical values for their fitness level
3. **Actionable**: Provide specific training recommendations based on CTL/ATL/TSB trends
4. **Progressive**: Track improvements over time, celebrate PRs and consistency
5. **Safety-First**: Watch for overtraining signals (high ATL, negative TSB), recommend recovery when needed
6. **CRITICAL**: For aggregation queries (average, total, calculate, all activities), use ALL provided activities in your calculations, not just the first few

FORMATTING STYLE:
- Start with an engaging, personalized greeting referencing their specific activity
- Use emojis strategically: 🚴‍♂️ (cycling), 🏃‍♂️ (running), 🏊‍♂️ (swimming), 💪 (strength), 🔥 (intensity), 📈 (progress), ⚡ (power), ❤️ (heart rate)
- Structure with clear sections using ### headers
- **Bold** key metrics and achievements
- Use bullet points for multiple insights
- Include "📊 **By the Numbers:**" section for key stats
- Add "🎯 **Training Focus:**" for recommendations
- End with 2-3 specific follow-up questions

SPECIAL CONSIDERATIONS:
- **VirtualRide = Indoor Training**: Acknowledge indoor trainer sessions, consider no GPS/weather data
- **Missing Data**: If no power/HR data, focus on duration, perceived effort, consistency
- **Aggregations**: For totals/averages, process ALL provided activities, show calculation transparency
- **No Activities Found**: Acknowledge this clearly, ask about manual uploads or sync issues
- **Privacy**: Never ask for personal info beyond what's in Strava data

CONVERSATION CONTINUITY:
- Build on previous messages in the conversation
- Remember what was discussed earlier
- Use pronouns naturally ("your morning ride" → "it" in follow-ups)
- Track goals or concerns mentioned across messages

Remember: You're their trusted training companion. Be encouraging but honest, data-driven but personable, expert but accessible."""
    
    def generate_conversational_response(
        self,
        user_query: str,
        relevant_activities: List[Dict[str, Any]],
        conversation_history: List[ChatMessage] = None
    ) -> str:
        """
        Generate a conversational response based on user query and relevant activities.
        
        Args:
            user_query: User's question or query
            relevant_activities: List of activities from vector search
            conversation_history: Previous conversation messages
            
        Returns:
            Conversational response from the LLM
        """
        try:
            # Prepare context from activities
            activities_context = self._format_activities_for_context(relevant_activities)
            
            # Build messages for the conversation
            messages = [
                {"role": "system", "content": self._create_system_prompt()}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Add current query with context
            user_message = f"""User query: {user_query}

Relevant activities from their Strava data:
{activities_context}

Please provide a helpful, conversational response based on this activity data."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def _format_activities_for_context(self, activities: List[Dict[str, Any]]) -> str:
        """Format activities data for LLM context."""
        if not activities:
            return "No relevant activities found in the database."
        
        context_parts = []
        # Use all activities provided, but limit to reasonable display size
        max_activities = min(len(activities), 50)  # Display up to 50 activities
        for i, activity in enumerate(activities[:max_activities], 1):
            metadata = activity.get('metadata', {})
            summary = activity.get('summary', '')
            score = activity.get('score', 0)
            
            # Extract key metrics
            activity_type = metadata.get('activity_type', 'unknown')
            start_date = metadata.get('start_date', '')
            distance = metadata.get('distance', 0)
            moving_time = metadata.get('moving_time', 0)
            elevation_gain = metadata.get('total_elevation_gain', 0)
            avg_heartrate = metadata.get('average_heartrate')
            avg_speed = metadata.get('average_speed', 0)
            
            # Format the activity info
            activity_info = [
                f"Activity {i} (Relevance: {score:.2f}):",
                f"  Type: {activity_type.title()}",
                f"  Date: {start_date}",
                f"  Summary: {summary}"
            ]
            
            # Add metrics if available
            if distance > 0:
                distance_km = distance / 1000
                activity_info.append(f"  Distance: {distance_km:.2f} km")
            
            if moving_time > 0:
                hours = moving_time // 3600
                minutes = (moving_time % 3600) // 60
                if hours > 0:
                    activity_info.append(f"  Duration: {hours}h {minutes}m")
                else:
                    activity_info.append(f"  Duration: {minutes}m")
            
            if elevation_gain > 0:
                activity_info.append(f"  Elevation Gain: {elevation_gain}m")
            
            if avg_heartrate:
                activity_info.append(f"  Average Heart Rate: {avg_heartrate} bpm")
            
            if avg_speed > 0:
                speed_kmh = avg_speed * 3.6
                activity_info.append(f"  Average Speed: {speed_kmh:.1f} km/h")
            
            context_parts.append("\n".join(activity_info))
        
        return "\n\n".join(context_parts)
    
    def generate_follow_up_questions(
        self,
        user_query: str,
        response: str,
        activities: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate relevant follow-up questions based on the conversation.
        
        Args:
            user_query: Original user query
            response: LLM response
            activities: Activities that were referenced
            
        Returns:
            List of follow-up questions
        """
        try:
            prompt = f"""Based on this fitness conversation, suggest 3 relevant follow-up questions the user might want to ask about their Strava activities:

User asked: {user_query}
Assistant responded: {response}

Available activity data includes: {len(activities)} activities with metrics like distance, pace, heart rate, elevation, etc.

Generate 3 specific, actionable follow-up questions that would help the user explore their fitness data further. Make them conversational and engaging.

Format as a simple list:
1. [question]
2. [question]
3. [question]"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.8
            )
            
            # Parse the response into a list
            content = response.choices[0].message.content
            questions = []
            
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.')) or line.startswith('-')):
                    # Remove numbering and clean up
                    question = line.split('.', 1)[-1].strip()
                    if question:
                        questions.append(question)
            
            return questions[:3]  # Ensure we return max 3 questions
            
        except Exception:
            # Return fallback questions if generation fails
            return [
                "What was my best performance this month?",
                "How has my training been trending lately?",
                "What activities should I focus on next?"
            ]
    
    def analyze_fitness_trends(
        self,
        activities: List[Dict[str, Any]],
        time_period: str = "month"
    ) -> str:
        """
        Generate insights about fitness trends from activity data.
        
        Args:
            activities: List of activities to analyze
            time_period: Time period for analysis (week, month, year)
            
        Returns:
            Trend analysis insights
        """
        try:
            # Prepare activities data for analysis
            activities_summary = self._summarize_activities_for_analysis(activities)
            
            prompt = f"""Analyze the following fitness activities and provide insights about trends and patterns over the past {time_period}:

{activities_summary}

Please provide:
1. Overall activity trends (frequency, types, progression)
2. Performance insights (improvements, consistency, areas for growth)
3. Training patterns (preferred activities, timing, intensity)
4. Specific recommendations for continued improvement
5. Celebration of achievements and milestones

Keep the analysis encouraging and actionable."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.6
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Failed to analyze trends: {str(e)}")
    
    def _summarize_activities_for_analysis(self, activities: List[Dict[str, Any]]) -> str:
        """Summarize activities for trend analysis."""
        if not activities:
            return "No activities available for analysis."
        
        # Group activities by type
        activity_types = {}
        total_distance = 0
        total_time = 0
        
        for activity in activities:
            metadata = activity.get('metadata', {})
            activity_type = metadata.get('activity_type', 'unknown')
            distance = metadata.get('distance', 0)
            moving_time = metadata.get('moving_time', 0)
            
            if activity_type not in activity_types:
                activity_types[activity_type] = {
                    'count': 0,
                    'total_distance': 0,
                    'total_time': 0
                }
            
            activity_types[activity_type]['count'] += 1
            activity_types[activity_type]['total_distance'] += distance
            activity_types[activity_type]['total_time'] += moving_time
            
            total_distance += distance
            total_time += moving_time
        
        # Create summary
        summary_parts = [
            f"Total activities analyzed: {len(activities)}",
            f"Total distance: {total_distance/1000:.1f} km",
            f"Total time: {total_time//3600:.1f} hours",
            "\nActivity breakdown:"
        ]
        
        for activity_type, stats in activity_types.items():
            summary_parts.append(
                f"  {activity_type.title()}: {stats['count']} activities, "
                f"{stats['total_distance']/1000:.1f} km, "
                f"{stats['total_time']//3600:.1f} hours"
            )
        
        return "\n".join(summary_parts)