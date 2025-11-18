import pickle

vectorizer, model = pickle.load(open('bank_model.pkl', 'rb'))

conversation_history = []
first_interaction = True

while True:
    if first_interaction:
        user_input = [input("Ask a bank related question: ").strip()]
        first_interaction = False
    else:
        user_input = [input("\nAny other queries? (Type 'exit' to quit): ").strip()]

    if user_input[0].lower() == "exit":
        print("Thank you for banking with us.")
        break
    if not user_input:
        print("Please enter a question: ")
        continue

    conversation_history.append(user_input)
    
    recent_convo = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history

    X = vectorizer.transform(user_input)
    print(model.predict(X))