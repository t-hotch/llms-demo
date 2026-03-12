'''Chat bot demo using Ollama and LangChain

Start the ollama server, and download the model:

$ ollama serve
$ ollama pull qwen2.5:3b

'''

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

model = 'qwen2.5:3b'

system_prompt = ('You are a helpful teaching assistant at an AI/ML boot camp.',
                'Answer questions in simple language with examples when possible.',
                'Answer in the style of a pirate and use nautical themed analogies.')

llm = ChatOllama(
    model=model,
    temperature=0.9
)

history = [SystemMessage(content=system_prompt)]

def main():

    while True:

        user_input = input('User: ')

        if user_input == 'exit':
            break

        history.append(HumanMessage(content=user_input))
        response = llm.invoke(history)
        history.append(response)

        print(f'\n{model}: {response.content}\n')


if __name__ == '__main__':

    main()