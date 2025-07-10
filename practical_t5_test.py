from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

class T5PracticalTester:
    def __init__(self):
        print("🔄 Loading T5 model and tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        print("✅ Model loaded successfully!\n")

    def generate_response(self, input_text, max_length=100):
        start_time = time.time()
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()
        return response, end_time - start_time

    def print_result(self, task, input_text, output, time_taken):
        print(f"\n{'='*50}")
        print(f"🎯 Task: {task}")
        print(f"📥 Input: {input_text}")
        print(f"📤 Output: {output}")
        print(f"⏱️ Time taken: {time_taken:.2f} seconds")
        print(f"{'='*50}")

    def run_practical_tests(self):
        tests = [
            {
                "task": "Code Documentation",
                "input": "translate English to Python docstring: This function calculates the factorial of a number using recursion"
            },
            {
                "task": "Text Summarization",
                "input": "summarize: Machine learning is a subset of artificial intelligence that involves training computer systems to learn from data without explicit programming. These systems can identify patterns, make decisions, and improve their performance over time through experience."
            },
            {
                "task": "Grammar Correction",
                "input": "grammar: he dont like to eat vegetables and dont drink milk"
            },
            {
                "task": "Question Generation",
                "input": "generate question: context: Python is a popular programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991."
            }
        ]

        for test in tests:
            output, time_taken = self.generate_response(test["input"])
            self.print_result(test["task"], test["input"], output, time_taken)

def main():
    try:
        print("🚀 Starting Practical T5 Model Tests...")
        tester = T5PracticalTester()
        tester.run_practical_tests()
        print("\n✨ All practical tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    main()
