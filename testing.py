from iteung import reply

while True:
    message = input("Kamu: ")
    return_message, status = reply.botReply(message)
    print(f"ITeung: {return_message}")