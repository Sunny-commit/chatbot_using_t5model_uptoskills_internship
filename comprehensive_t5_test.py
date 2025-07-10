from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class T5Tester:
    def __init__(self):
        print("Initializing T5 model and tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        print("Initialization complete!\n")

    def generate_response(self, input_text, max_length=50):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run_translation_tests(self):
        print("\n=== Translation Tests ===")
        tests = [
            "translate English to German: The weather is beautiful today.",
            "translate English to French: I love programming.",
            "translate English to Spanish: How are you doing?",
            "translate English to German: Can you help me with this problem?"
        ]
        
        for test in tests:
            print(f"\nInput: {test}")
            print(f"Output: {self.generate_response(test)}")

    def run_summarization_tests(self):
        print("\n=== Summarization Tests ===")
        texts = [
            "summarize: The T5 model is a transformer-based machine learning model that can perform various natural language processing tasks. It was developed by Google Research and has shown impressive results in translation, summarization, and question answering tasks.",
            "summarize: Artificial intelligence has transformed many industries including healthcare, finance, and transportation. Machine learning models can now predict diseases, detect fraud, and drive cars autonomously.",
        ]
        
        for text in texts:
            print(f"\nInput: {text}")
            print(f"Output: {self.generate_response(text, max_length=50)}")

    def run_qa_tests(self):
        print("\n=== Question Answering Tests ===")
        questions = [
            "question: What is the capital of France? context: Paris is the capital and largest city of France.",
            "question: Who invented the telephone? context: Alexander Graham Bell is credited with inventing the first practical telephone.",
            "question: What is Python? context: Python is a high-level, interpreted programming language created by Guido van Rossum."
        ]
        
        for question in questions:
            print(f"\nInput: {question}")
            print(f"Output: {self.generate_response(question)}")

def main():
    try:
        tester = T5Tester()
        
        # Run all tests
        tester.run_translation_tests()
        tester.run_summarization_tests()
        tester.run_qa_tests()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    main()
