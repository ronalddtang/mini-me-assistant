while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        print("Exiting the assistant...")
        break
    print(f"Assistant: You said {user_input}")
