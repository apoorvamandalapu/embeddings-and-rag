import rag_service

if __name__ == "__main__":
    print("Search is ready. Type question or 'exit' to quit.")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        try:
            print("Searching for relevant information...")
            context = rag_service.search_db(query, top_k=3)
            
            print("\nGenerating a comprehensive answer...")
            answer = rag_service.generate_answer_from_context(query, context)
            
            print("\nFinal Answer:")
            print(answer)
        
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check your API keys and code.")