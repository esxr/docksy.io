//------------------------------------------------------------
// LOADING, SPLITTING, AND STORAGE
//------------------------------------------------------------

/**
 * Specify a Document loader.
 */
// Document loader

// import OPENAI API KEY from .env file
import dotenv from "dotenv";
dotenv.config();

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
);
const data = await loader.load();

/**
 * Split the Document into chunks for embedding and vector storage.
 */
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 0,
});

const splitDocs = await textSplitter.splitDocuments(data);

/**
 * Embed and store the splits in a vector database (for demo purposes we use an unoptimized, in-memory example)
 */
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const embeddings = new OpenAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

//------------------------------------------------------------
// RETRIEVAL
//------------------------------------------------------------

/**
 * Retrieve relevant splits for any question using similarity_search.
 */
const relevantDocs = await vectorStore.similaritySearch("What is task decomposition?");
console.log(relevantDocs.length);
// 4

//------------------------------------------------------------
// QUESTION AND ANSWER
//------------------------------------------------------------

/**
 * Distill the retrieved documents into an answer using an LLM (e.g., gpt-3.5-turbo) with RetrievalQA chain.
 */
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

var model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
var chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

const response = await chain.call({
  query: "What is task decomposition?"
});
console.log(response);

/*
  {
    text: 'Task decomposition refers to the process of breaking down a larger task into smaller, more manageable subgoals. By decomposing a task, it becomes easier for an agent or system to handle complex tasks efficiently. Task decomposition can be done through various methods such as using prompting or task-specific instructions, or through human inputs. It helps in planning and organizing the steps required to complete a task effectively.'
  }
*/

//------------------------------------------------------------
// CHAT
//------------------------------------------------------------

/**
 * To keep chat history, we use a variant of the previous chain called a ConversationalRetrievalQAChain. First, specify a Memory buffer to track the conversation inputs / outputs.
 */
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";

const memory = new BufferMemory({
  memoryKey: "chat_history",
  returnMessages: true,
});

/**
 * Next, we initialize and call the chain:
 */
var model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
var chain = ConversationalRetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
  memory
});

const result = await chain.call({
  question: "What are some of the main ideas in self-reflection?"
});
console.log(result);

/*
{
  text: 'Some main ideas in self-reflection include:\n' +
    '\n' +
    '1. Iterative Improvement: Self-reflection allows autonomous agents to improve by continuously refining past action decisions and correcting mistakes.\n' +
    '\n' +
    '2. Trial and Error: Self-reflection plays a crucial role in real-world tasks where trial and error are inevitable. It helps agents learn from failed trajectories and make adjustments for future actions.\n' +
    '\n' +
    '3. Constructive Criticism: Agents engage in constructive self-criticism of their big-picture behavior to identify areas for improvement.\n' +
    '\n' +
    '4. Decision and Strategy Refinement: Reflection on past decisions and strategies enables agents to refine their approach and make more informed choices.\n' +
    '\n' +
    '5. Efficiency and Optimization: Self-reflection encourages agents to be smart and efficient in their actions, aiming to complete tasks in the least number of steps.\n' +
    '\n' +
    'These ideas highlight the importance of self-reflection in enhancing performance and guiding future actions.'
}
*/

/**
 * The Memory buffer has context to resolve "it" ("self-reflection") in the below question.
 */
const followupResult = await chain.call({
    question: "How does the Reflexion paper handle it?"
  });
  console.log(followupResult);
  
  /*
  {
    text: "The Reflexion paper introduces a framework that equips agents with dynamic memory and self-reflection capabilities to improve their reasoning skills. The approach involves showing the agent two-shot examples, where each example consists of a failed trajectory and an ideal reflection on how to guide future changes in the agent's plan. These reflections are then added to the agent's working memory as context for querying a language model. The agent uses this self-reflection information to make decisions on whether to start a new trial or continue with the current plan."
  }
  */

