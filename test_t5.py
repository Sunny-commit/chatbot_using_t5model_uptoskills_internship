from transformers import T5ForConditionalGeneration, T5Tokenizer

def test_t5():
    try:
        print("Loading T5 tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        print("Loading T5 model...")
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        
        print("Testing with a sample input...")
        input_text = "translate English to German: Hello, how are you?"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        outputs = model.generate(input_ids, max_length=50)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nInput: {input_text}")
        print(f"Output: {translation}")
        print("\nT5 model is working correctly!")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_t5()
