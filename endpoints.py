from flask import Response, stream_with_context, request, Flask
from flask_cors import CORS
from markupsafe import escape
from csv import reader
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

app = Flask(__name__)
CORS(app)

conversations = []


@app.route("/courses")
def get_courses():
    csvfile = open("data/courses.csv", encoding="utf-8")
    csvreader = reader(csvfile)
    next(csvreader)
    res = []
    for row in csvreader:
        course_info = {
            "name": row[0],
            "course_id": row[1],
            "credits": row[2]
        }
        res.append(course_info)
    return res


def find_conv(conv_id):
    for conv in conversations:
        if conv["session"] == conv_id:
            return conv
    return None


@app.route("/conv/<conv_id>", methods=["POST"])
def talk(conv_id):
    conv = find_conv(conv_id)
    if not conv:
        conv = {"session": conv_id, "messages": []}
        conversations.append(conv)

    user_input = request.get_json()
    conv["messages"].append({"role": "user", "content": user_input["message"]})

    def generate():
        stream = client.responses.create(
            model="openai/gpt-oss-20b",
            input=conv["messages"],
            stream=True,
            # print() doesn't like emojis
            # instructions="No unicode emojis in responses."
        )

        assistant_output_messages = ""
        for chunk in stream:
            text = chunk.delta if chunk.type == "response.output_text.delta" else None
            if text:
                assistant_output_messages += text
                yield text

        answer = {
            "role": "assistant",
            "content": assistant_output_messages
        }
        conv["messages"].append(answer)
    return Response(stream_with_context(generate()), mimetype="text/plain")
