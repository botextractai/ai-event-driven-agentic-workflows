# Event-Driven Agentic Workflows with Human-in-the-Loop

LlamaIndex agentic document workflows are agent-based applications that can be used to automate end-to-end processing workflows. Workflows prescribe data flows to agents.

While Retrieval-Augmented Generation (RAG) systems answer simple questions about data, agentic workflows can be built on top of RAG to help process input documents in more sophisticated ways.

In this example, an agent identifies key information and retrieves relevant parts using RAG. The agent then combines the collected information into a structured output that fits the target format.

This example builds an agent that processes a resume to fill in a job application form.

![alt text](https://github.com/user-attachments/assets/dcf1cf01-9798-4f04-a05c-2b94208e022f "Workflow Flowchart")

## Required API keys for this example

This example requires API keys from both OpenAI and LLamaCloud.

You need to insert these 2 pieces of information into the `.env.example` file and then rename this file to just `.env` (remove the ".example" ending).

1. [Get your OpenAI API key here](https://platform.openai.com/login).
2. [Get your free LlamaCloud API key here](https://cloud.llamaindex.ai/login) for up to 1000 LlamaParse pages per day.

## Event-based workflows

LlamaIndex workflows are an event-driven architecture that can be used to create an agent. The agent's logic can be encapsulated in a chain of steps, where each steps emits events to trigger further steps. Multiple steps are created by defining custom events that can be emitted by steps and trigger other steps.

Under the hood, LlamaIndex workflows are regular Python classes. Workflows are defined as a series of steps, each of which receives and emits certain classes of events.

Workflows can include loops, branching, and concurrent executions. For concurrent executions, workflows can be instructed to wait until all or a set number of executions concluded.

Workflows are asynchronous (async) by default.

## The example workflow

The example workflow first uses simple RAG to parse a resume using LlamaParse and load it into a vector store. It then uses the agent to run basic queries against the document.

The RAG embedding uses OpenAI's `text-embedding-3-small` model. The querying uses OpenAI's `gpt-4o-mini` Large Language Model (LLM).

The indexes in this example are persisted to disk. In a production setting with more documents and users, this might be changed to a hosted vector store.

In this example workflow, an application form then gets read and converted into a list of fields that need to be filled in. This list of fields is then returned as a JSON object. The `Query_Event` is fired off for each of the fields. These queries are executed concurrently to save time.

## Human-in-the-Loop

The workflow makes it easy to iterate on answers to the application form by getting feedback on answers from a human operator and re-answering them when necessary.

`InputRequiredEvent` and `HumanResponseEvent` are LlamaIndex special events designed to allow a human user to exit the workflow, or to get feedback back into it.

After every human feedback, the LLM parses the feedback and decides whether it means that it should terminate and output the final response, or if it needs to loop back and do more work.

When executing this example, it will generate a first response. It will then pause to ask:

> How does this look? Give me any feedback you have on any of the answers.

You will have two choices.

1. You can provide feedback to enhance the response by looping. For example, you can enter this text to add the name of the university to the degree:

   ```
   For the degree, add the institution
   ```

   You will then see the reply

   > LLM says the verdict was FEEDBACK

   while the response gets enhanced.

2. You can provide feedback that the response is good enough, in which case the response will be considered final.

   You can simply answer with something positive like:

   ```
   Well done!
   ```

   You will then see the reply

   > LLM says the verdict was OKAY

   and get the final response.
