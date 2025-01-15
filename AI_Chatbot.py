from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate chatbot responses
def get_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("message", "")
        if not user_input:
            return "<h3>Please enter a message!</h3>"
        ai_response = get_response(user_input)
        return f"<h3>Chatbot Response: {ai_response}</h3><a href='/chat'>Back</a>"

    # Display a simple HTML form for GET requests
    return '''
    <form action="/chat" method="post">
        <label for="message">Enter your message:</label><br>
        <input type="text" id="message" name="message" required><br><br>
        <button type="submit">Send</button>
    </form>
    '''

if __name__ == "__main__":
    app.run(port=5000)