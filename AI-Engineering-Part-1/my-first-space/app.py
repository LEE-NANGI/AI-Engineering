import gradio as gr
import spaces

@spaces.GPU
def respond_to_user_input(message, history):
    # Here you would implement the logic to process the user input and generate a response
    # For demonstration purposes, we'll just echo the message back
    response = f"You said: {message} \
        \n I am learning AI Engineering"
    return response

gr.ChatInterface(fn=respond_to_user_input).launch()
    