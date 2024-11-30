import { fileURLToPath } from "url";
import path from "path";

const jsonSchemaGrammar: Readonly<{
  type: string;
  properties: {
    user: {
      type: string;
    };
    content: {
      type: string;
    };
  };
}> = {
  type: "object",
  properties: {
    user: {
      type: "string",
    },
    content: {
      type: "string",
    },
  },
};

interface QueuedMessage {
  context: string;
  temperature: number;
  stop: string[];
  max_tokens: number;
  frequency_penalty: number;
  presence_penalty: number;
  useGrammar: boolean;
  resolve: (value: any | string | PromiseLike<any | string>) => void;
  reject: (reason?: any) => void;
}

import OpenAI from 'openai';

class LlamaService {
  private static instance: LlamaService | null = null;
  private openai: OpenAI | null = null;
  private modelName: string = 'llama3.2';
  private embeddingModelName: string = 'mxbai-embed-large';
  private messageQueue: QueuedMessage[] = [];
  private isProcessing: boolean = false;

  private constructor() {
    this.openai = new OpenAI({
      baseURL: 'http://localhost:11434/v1',
      apiKey: 'ollama', // Required but unused
    });
  }

  public static getInstance(): LlamaService {
    if (!LlamaService.instance) {
      LlamaService.instance = new LlamaService();
    }
    return LlamaService.instance;
  }

  async initializeModel() {
    // No initialization required for Ollama
    console.log("Ollama model initialized.");
  }

  async checkModel() {
    // No model checking required for Ollama
    console.log("Ollama model check passed.");
  }

  async deleteModel() {
    // No model deletion required for Ollama
    console.log("Ollama model deletion not applicable.");
  }

  async queueMessageCompletion(
    context: string,
    temperature: number,
    stop: string[],
    frequency_penalty: number,
    presence_penalty: number,
    max_tokens: number,
  ): Promise<any> {
    console.log("Queueing message completion with Ollama");
    return new Promise((resolve, reject) => {
      this.messageQueue.push({
        context,
        temperature,
        stop,
        frequency_penalty,
        presence_penalty,
        max_tokens,
        useGrammar: true,
        resolve,
        reject,
      });
      this.processQueue();
    });
  }

  async queueTextCompletion(
    context: string,
    temperature: number,
    stop: string[],
    frequency_penalty: number,
    presence_penalty: number,
    max_tokens: number,
  ): Promise<string> {
    console.log("Queueing text completion with Ollama");
    return new Promise((resolve, reject) => {
      this.messageQueue.push({
        context,
        temperature,
        stop,
        frequency_penalty,
        presence_penalty,
        max_tokens,
        useGrammar: false,
        resolve,
        reject,
      });
      this.processQueue();
    });
  }

  private async processQueue() {
    if (this.isProcessing || this.messageQueue.length === 0) {
      return;
    }

    this.isProcessing = true;

    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        try {
          console.log("Processing message with Ollama");
          const response = await this.getCompletionResponse(
            message.context,
            message.temperature,
            message.stop,
            message.frequency_penalty,
            message.presence_penalty,
            message.max_tokens,
          );
          message.resolve(response);
        } catch (error) {
          message.reject(error);
        }
      }
    }

    this.isProcessing = false;
  }

  private async getCompletionResponse(
    context: string,
    temperature: number,
    stop: string[],
    frequency_penalty: number,
    presence_penalty: number,
    max_tokens: number,
  ): Promise<any | string> {
    if (!this.openai) {
      throw new Error("Ollama API not initialized.");
    }
  
    const completion = await this.openai.chat.completions.create({
      model: this.modelName,
      messages: [{ role: 'user', content: context }],
      temperature,
      max_tokens,
      stop,
      frequency_penalty,
      presence_penalty,
    });
  
    let content = completion.choices[0].message.content;
    
    // Fix unquoted action values
    content = content.replace(/("action":\s*)(\w+)/, (match, p1, p2) => {
      return `${p1}"${p2}"`;
    });
  
    return content;
  }

  async getEmbeddingResponse(input: string): Promise<number[] | undefined> {
    if (!this.openai) {
      throw new Error("Ollama API not initialized.");
    }

    try {
      const embeddingResponse = await this.openai.embeddings.create({
        model: this.embeddingModelName,
        input,
      });

      return embeddingResponse.data[0].embedding;
    } catch (error) {
      console.error("Error getting embedding response:", error);
      return undefined;
    }
  }
}

export default LlamaService;
