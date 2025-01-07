import os
import socket
from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Callable, Literal

from langfuse import Langfuse
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
PD_CORE_VERSION = os.getenv("PD_CORE_VERSION", "unknown")
PD_CORE_RELEASE = os.getenv("PD_CORE_RELEASE", "unknown")  # e.g. Nov 16, 2024 18:53
PD_ENVIRONMENT = os.getenv("PD_ENVIRONMENT", "dev") # "prod" or "dev"

PD_CORE_USER_ID = os.getenv("PD_CORE_USER_ID", socket.gethostname())
PD_CORE_SESSION_ID = os.getenv("PD_CORE_SESSION_ID", "unspecified")  # Equal to the job-id of the backend


class Agent(ABC):
    """
    Every Agent has its own trace, which records all operations.
    """

    def __init__(self, name: str, model: str, temperature: float = 0.0, seed: int = 0, tags: list[str] | None = None, top_p: float = 0.1):
        """
        @param name: Agent Name, e.g. "RuleApply" or "PaperScore"
        """
        self.name = name
        # self.client = OpenAI(
        #     base_url="https://api.ai-gaochao.cn/v1",
        #     api_key="sk-TUe0juCu1185CZQoCc33035718E642538f7706904b63C1F5",
        # )
        self.client = OpenAI(
            base_url="https://api.openai.com/v1",
            organization='org-Bis7Azo6YLHnUAacSoa4OVr0',
            project='proj_ABbJAUN09IIJrt8M7bkomA7T',
            api_key="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")
        if 'o1' not in model:
            self.temperature = temperature  # OpenAI Temperature
            self.top_p = top_p              # OpenAI Top P
        self.seed = seed
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-40c1e42e-3e3c-4a6d-88fe-acb2967ae702"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-68f89f66-6f42-4d09-be12-8a591f5bca58"),
            host=os.getenv("LANGFUSE_HOST", "https://pd-trace.xtra.science")
        )

        self.trace = self.langfuse.trace(
            name=self.name,
            tags=(tags or []) + [f"env: {PD_ENVIRONMENT}"],
            user_id=PD_CORE_USER_ID,
            session_id=PD_CORE_SESSION_ID,
            version=PD_CORE_VERSION,
            release=PD_CORE_RELEASE,
            input="Please check the Observation for GENERATION",
            output="Please check the Observation for GENERATION",
            metadata={
                "hint": "Please check the Observation for GENERATION",
            }
        )
        self.model_name = model
        if 'gpt' not in self.model_name and 'o1' not in self.model_name:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype="auto")
        else:
            self.model = model


    def run(
            self, *,
            prompt: str | object,
            stream_callback: Callable[[str], None] | None = None,
            # model: str = "gpt-4o-mini",
            response_format: Literal["text", "json_object"] = "text",
            tags: list[str] | None = None,
            metadata: dict | None = None,
            debug: bool = False
    ) -> str:
        """
        Generate a response from the given prompt.
        @param prompt: It can be a string or a list of messages.
        @param stream_callback: A callback function that will be called with each new token in the response stream.
        @param model: The model to use for the generation. Default is "gpt-4o-mini"
        @param tags: Tags for the observation (not the trace).
        @param metadata: tracing only. you can put any key-value pairs in it.
        """
        if 'o1' in self.model_name:
            trace_generate = self.trace.generation(
                name="generate_response",
                input=prompt,
                metadata={
                    **(metadata or {}),
                    "seed": self.seed,
                },
                tags=(tags or []) + [f"env: {PD_ENVIRONMENT}"],
                user_id=PD_CORE_USER_ID,
                session_id=PD_CORE_SESSION_ID,
                model=self.model,
                start_time=datetime.now(),
                level="DEBUG" if debug else "DEFAULT"
            )
        else:
            trace_generate = self.trace.generation(
                name="generate_response",
                input=prompt,
                metadata={
                    **(metadata or {}),
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "seed": self.seed,
                },
                tags=(tags or []) + [f"env: {PD_ENVIRONMENT}"],
                user_id=PD_CORE_USER_ID,
                session_id=PD_CORE_SESSION_ID,
                model=self.model,
                start_time=datetime.now(),
                level="DEBUG" if debug else "DEFAULT"
            )            
        print("model_name:", self.model_name)
        if 'gpt' in self.model_name or 'o1' in self.model_name:

            if 'gpt' in self.model_name:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt,
                    stream=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.seed,
                    response_format={"type": response_format}
                )
            else:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt,
                    stream=True,
                    seed=self.seed,
                    response_format={"type": response_format}
                )

            collected = []
            for s in stream:
                # print("s:",s)
                try:
                    content = s.choices[0].delta.content

                    if content is not None:
                        collected.append(content)
                        if stream_callback is not None:
                            stream_callback(content)
                except Exception as e:
                    break
            output = "".join(collected)

            self.trace.update(output=output)  # update the trace with the final output
            trace_generate.update(output=output, end_time=datetime.now())
            return output
        else:


            print(len(prompt))
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=5000
            )
            print('##########################')
            print(generated_ids)
            print(generated_ids.shape)
            print(generated_ids.shape[0])
            print('##########################')
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if stream_callback is not None:
                for chunk in response:
                    stream_callback(chunk)

            self.trace.update(output=response)  # 更新 trace 中的最终输出
            trace_generate.update(output=response, end_time=datetime.now())
            return response

    def __del__(self):
        self.langfuse.flush()
