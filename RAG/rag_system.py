import asyncio
import numpy as np
import os
from ollama import AsyncClient
from sentence_transformers import SentenceTransformer
from server.config import AppConfig
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



class GemmaChatContext:
    """https://ai.google.dev/gemma/docs/pytorch_gemma"""

    USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
    MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.history = []

    def add_user_query(self, query):
        self.history.append(GemmaChatContext.USER_CHAT_TEMPLATE.format(prompt=query))

    def add_model_response(self, resp):
        self.history.append(GemmaChatContext.MODEL_CHAT_TEMPLATE.format(prompt=resp))

    def reset(self):
        self.history = []

    def generate_prompt(self, knowledge_context = None):

        ctx_len = self.cfg.llm.context_length
        if ctx_len > 0:
            self.history = self.history[-ctx_len * 2 :]
        else:
            self.history = self.history[-1:]
        context = "".join(self.history)

        system_message = self.cfg.llm.system_message
        system_message = system_message if isinstance(system_message, str) else ""
        system_message = system_message.strip()
        sys_prompt = ""
        if len(system_message) > 0:
            sys_prompt = GemmaChatContext.USER_CHAT_TEMPLATE.format(
                prompt=system_message
            )

        knowledge_context = f"context: {knowledge_context} \n"

        return sys_prompt + context + knowledge_context + "<start_of_turn>model\n"


class VectorKnowledgeBase:
    def __init__(self, file_path: str, model_name: str = "all-MiniLM-L6-v2"):
        with open(file_path, "r", encoding="utf-8") as f:
            self.chunks = [line.strip() for line in f if line.strip()]
        print(f"🔄 Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        print("🧠 Encoding knowledge base...")
        self.chunk_embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        print(f"✅ Knowledge base ready: {len(self.chunks)} chunks loaded.")

    def retrieve(self, query: str) -> str:
        query_emb = self.embedder.encode(query, convert_to_numpy=True)
        sims = np.dot(self.chunk_embeddings, query_emb) / (
            np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        best_idx = int(np.argmax(sims))
        return self.chunks[best_idx]




class RAGBot:
    def __init__(self, cfg: AppConfig, llm: AsyncClient):
        self.cfg = cfg
        self.llm = llm
        self.knowledge_base = VectorKnowledgeBase(cfg.llm.knowledge_file)
        self.chat_context = GemmaChatContext(cfg)

    async def process_query(self, query: str) -> str:
        self.chat_context.add_user_query(query)
        context = self.knowledge_base.retrieve(query)
        response = await self.generate_answer_with_context(context)
        self.chat_context.add_model_response(response)
        return response

    async def generate_answer_with_context(self,knonwledge_context: str) -> str:

        prompt = self.chat_context.generate_prompt(knonwledge_context)

        resp = await self.llm.generate(
            model=self.cfg.llm.model,
            prompt=prompt,
            options={"temperature": self.cfg.llm.temperature,
                "top_k": self.cfg.llm.top_k,
                "top_p": self.cfg.llm.top_p,}
        )
        return resp.get("response", "").strip()



async def run_interactive_mode(bot: RAGBot):
    print("🧠 RAGBot ready! Ask about:")
    print(" • Vending machine")
    print(" • Shishkin's painting")
    print(" • Next train to Odintsovo")
    print(" • Nearest restroom")
    print(" • Museum tour start time")
    print(" • Cloakroom location")
    print("\nType 'exit' to quit.\n")

    while True:
        try:
            query = input("💬 You: ").strip()
            if not query or query.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break
            response = await bot.process_query(query)
            print(f"🤖 Bot: {response}\n")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    bot = RAGBot(model_name="gemma3:270m", knowledge_file="knowledge.txt")
    asyncio.run(run_interactive_mode(bot))