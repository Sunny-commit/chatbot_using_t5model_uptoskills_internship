import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
from pathlib import Path
import traceback

class T5ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = 't5-small'
        self.is_model_loaded = False
        self.error_message = None

    def load_model(self):
        try:
            with st.spinner("Loading T5 model and tokenizer..."):
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.is_model_loaded = True
                return True
        except Exception as e:
            self.error_message = f"Error loading model: {str(e)}"
            return False

    def generate_response(self, input_text, max_length=150):
        try:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
            outputs = self.model.generate(input_ids, max_length=max_length)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True), None
        except Exception as e:
            return None, f"Error generating response: {str(e)}"

def main():
    st.set_page_config(
        page_title="T5 Model Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            color: #155724;
            margin: 1rem 0;
        }
        .error-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            color: #721c24;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🤖 T5 Model Interactive Demo")
    st.markdown("---")

    # Initialize session state
    if 't5_handler' not in st.session_state:
        st.session_state.t5_handler = T5ModelHandler()

    # Model loading status
    if not st.session_state.t5_handler.is_model_loaded:
        if st.button("Load T5 Model"):
            if st.session_state.t5_handler.load_model():
                st.success("✅ Model loaded successfully!")
            else:
                st.error(st.session_state.t5_handler.error_message)
    else:
        st.success("✅ Model is loaded and ready to use!")

    # Task selection
    if st.session_state.t5_handler.is_model_loaded:
        st.markdown("## 📝 Select Task")
        task = st.selectbox(
            "Choose a task",
            ["Translation", "Summarization", "Grammar Correction", "Question Answering", "Custom Input"]
        )

        # Input section
        st.markdown("## 📥 Input")
        if task == "Translation":
            target_lang = st.selectbox("Select target language", ["German", "French", "Spanish", "Italian"])
            text = st.text_area("Enter text to translate")
            if text:
                input_text = f"translate English to {target_lang}: {text}"
        elif task == "Summarization":
            text = st.text_area("Enter text to summarize")
            if text:
                input_text = f"summarize: {text}"
        elif task == "Grammar Correction":
            text = st.text_area("Enter text for grammar correction")
            if text:
                input_text = f"grammar: {text}"
        elif task == "Question Answering":
            context = st.text_area("Enter context")
            question = st.text_input("Enter question")
            if context and question:
                input_text = f"question: {question} context: {context}"
        else:  # Custom Input
            input_text = st.text_area("Enter your custom input with prefix (e.g., 'translate English to German: Hello')")

        # Generate response
        if st.button("Generate Response", key="generate"):
            if 'input_text' in locals() and input_text.strip():
                with st.spinner("Generating response..."):
                    start_time = time.time()
                    response, error = st.session_state.t5_handler.generate_response(input_text)
                    time_taken = time.time() - start_time

                    if error:
                        st.error(error)
                    else:
                        st.markdown("## 📤 Output")
                        st.success(response)
                        st.info(f"⏱️ Time taken: {time_taken:.2f} seconds")
            else:
                st.warning("Please provide input text")

        # Add some useful examples
        with st.expander("📚 Example Inputs"):
            st.markdown("""
            **Translation Example:**
            - "I love artificial intelligence"
            
            **Summarization Example:**
            - "The Internet of Things (IoT) refers to the network of physical objects embedded with sensors..."
            
            **Grammar Correction Example:**
            - "she dont know where is the book"
            
            **Question Answering Example:**
            - Context: "Python was created by Guido van Rossum in 1991"
            - Question: "Who created Python?"
            """)

    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and 🤗 Transformers")

if __name__ == "__main__":
    main()
