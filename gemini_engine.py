import os
import google.generativeai as genai


class ShelfSenseGemini:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        genai.configure(
            api_key=api_key
        )

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )


    def ask(
        self,
        question,
        detected_objects,
        rag_knowledge
    ):

        object_names = [
            obj["name"]
            for obj in detected_objects
        ]

        prompt = f"""
You are ShelfSense AI, an intelligent
computer-vision assistant.

You are analyzing the CURRENT IMAGE detected
by the ShelfSense system.

Detected objects:
{object_names}

Knowledge retrieved from ShelfSense RAG:
{rag_knowledge}

User question:
{question}

Instructions:

1. Answer specifically about the current scene.
2. Use the detected objects as visual context.
3. Use the RAG knowledge when relevant.
4. Do not invent objects that were not detected.
5. If the question is about organization,
   give practical recommendations.
6. If information is insufficient, clearly say so.
7. Keep the answer clear and useful.
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text