from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

class InteractiveT5:
    def __init__(self):
        print("🔄 Loading T5 model and tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        print("✅ Model loaded successfully!\n")

    def generate_response(self, input_text, max_length=150):
        start_time = time.time()
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        time_taken = time.time() - start_time
        return response, time_taken

    def print_result(self, task, input_text, output, time_taken):
        print(f"\n{'='*60}")
        print(f"🎯 Task: {task}")
        print(f"📥 Input: {input_text}")
        print(f"📤 Output: {output}")
        print(f"⏱️ Time taken: {time_taken:.2f} seconds")
        print(f"{'='*60}\n")

    def get_task_input(self, task):
        if task == "1":
            lang = input("Enter target language (e.g., German, French, Spanish): ")
            text = input("Enter text to translate: ")
            return f"translate English to {lang}: {text}"
        elif task == "2":
            return f"summarize: {input('Enter text to summarize: ')}"
        elif task == "3":
            return f"grammar: {input('Enter text for grammar correction: ')}"
        elif task == "4":
            context = input("Enter context: ")
            return f"question: {input('Enter question: ')} context: {context}"
        elif task == "5":
            return input("Enter your custom input with prefix (e.g., 'translate English to German: Hello'): ")
        return None

def main():
    try:
        t5 = InteractiveT5()
        
        while True:
            print("\n🔥 Available Tasks:")
            print("1. Translation")
            print("2. Text Summarization")
            print("3. Grammar Correction")
            print("4. Question Answering")
            print("5. Custom Input")
            print("6. Exit")
            
            choice = input("\nSelect a task (1-6): ").strip()
            
            if choice == "6":
                print("\n👋 Thank you for using the Interactive T5 Model!")
                break
                
            if choice not in ["1", "2", "3", "4", "5"]:
                print("❌ Invalid choice. Please select 1-6.")
                continue
                
            task_names = {
                "1": "Translation",
                "2": "Summarization",
                "3": "Grammar Correction",
                "4": "Question Answering",
                "5": "Custom Task"
            }
            
            input_text = t5.get_task_input(choice)
            if input_text:
                output, time_taken = t5.generate_response(input_text)
                t5.print_result(task_names[choice], input_text, output, time_taken)
            
            retry = input("\nTry another input? (y/n): ").strip().lower()
            if retry != 'y':
                print("\n👋 Thank you for using the Interactive T5 Model!")
                break
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
