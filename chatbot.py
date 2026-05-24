from dotenv import load_dotenv
import os
from google import genai
import streamlit as st

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

def generate_response(
    user_message,
    assessment_context,
    chat_history
):

    history_text = ""

    for msg in chat_history:

        history_text += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    assessment_summary = f"""

        Gender:
        {assessment_context.get('Gender','Unknown')}

        Occupation:
        {assessment_context.get('Occupation','Unknown')}

        Country:
        {assessment_context.get('Country','Unknown')}

        Family History:
        {assessment_context.get('FamilyHistory','Unknown')}

        Mental Health History:
        {assessment_context.get('MentalHealthHistory','Unknown')}

        Increasing Stress:
        {assessment_context.get('IncreasingStress','Unknown')}

        Mood Swings:
        {assessment_context.get('MoodSwings','Unknown')}

        Social Weakness:
        {assessment_context.get('SocialWeakness','Unknown')}

        Work Interest:
        {assessment_context.get('WorkInterest','Unknown')}

        Stress Score:
        {assessment_context.get('StressScore','Unknown')}

        Isolation Score:
        {assessment_context.get('IsolationScore','Unknown')}

        Prediction:
        {assessment_context.get('Prediction','Unknown')}

        Confidence:
        {assessment_context.get('Confidence','Unknown')}
        """
    prompt = f"""

            You are a supportive mental health assistant.
            Do not answer any other questions other than mental wellbeing and reject polietly.

            Use the user's assessment results to provide
            personalized guidance.

            Do not diagnose diseases or medications.

            If the user has not completed an assessment,
            suggest taking the assessment first.
            
            Suggest professional help if needed.

            USER ASSESSMENT

            {assessment_summary}

            CHAT HISTORY

            {history_text}

            CURRENT MESSAGE

            {user_message}
            
            Assessment Usage:

            - Use the user's assessment results when relevant.
            - Do not repeatedly list all weaknesses in every response.
            - Do not mention assessment findings unless they help answer the user's question.

            Recommendations:

            - If the user asks for solutions, improvement strategies, recommendations, tips, coping mechanisms, or asks how to improve a score, provide personalized suggestions based on their assessment results.

            Examples:

            If Mood Swings = High:
            - Suggest improving sleep consistency.
            - Suggest stress-management techniques.
            - Suggest journaling or emotional tracking.

            If Social Weakness = High:
            - Suggest gradually increasing social interactions.
            - Suggest reconnecting with trusted friends or family.

            If Isolation Score is elevated:
            - Suggest spending time outdoors.
            - Suggest participating in community or group activities.

            If Increasing Stress = Yes:
            - Suggest workload prioritization.
            - Suggest mindfulness exercises.
            - Suggest regular breaks and physical activity.

            Feature Questions:

            - If the user asks about a report metric, graph, score, confidence value, prediction, or dashboard feature, explain the meaning clearly.
            - When explaining report features, do not unnecessarily mention the user's weaknesses or risk factors.
            - Focus on explaining the feature itself.

            Important:

            - Only provide personalized risk-factor recommendations when the user explicitly asks for help, solutions, improvements, recommendations, or coping strategies.
            - Avoid repeatedly reminding users about their risk factors.

            """
    response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
            )
    return response.text