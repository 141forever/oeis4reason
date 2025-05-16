import json
from asyncio import Semaphore, gather
from contextlib import asynccontextmanager
from itertools import cycle

from openai import AsyncOpenAI, OpenAIError
from tqdm.asyncio import tqdm


def load_jsonl(path):
    with open(path, "r", encoding="utf8") as f:
        return [json.loads(_, strict=False) for _ in f]


def save_jsonl(data, path):
    with open(path, "w", encoding="utf8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


@asynccontextmanager
async def null_async_context():
    try:
        yield
    finally:
        return


class Agent:
    def __init__(self, apikeys: list[str] | str, **openai_args):
        if isinstance(apikeys, str):
            apikeys = [apikeys]
        self.apikeys = cycle(apikeys)
        self.openai_args = openai_args
        self.client = AsyncOpenAI(api_key=next(self.apikeys), **self.openai_args)

    async def _request_gpt(
        self,
        messages: list[dict[str, str]],
        request_kwargs: dict,
        sem: Semaphore | None = None,
        pbar: tqdm | None = None,
        ttl=3,
    ) -> str:
        # async with sem or null_async_context():
        try:
            response = await self.client.chat.completions.create(
                **request_kwargs, messages=messages
            )
            result = response.choices[0].message.content
        except OpenAIError as e:
            if ttl > 0:
                return await self._request_gpt(messages, request_kwargs, sem, pbar, ttl - 1)
            else:
                result = ""
        if pbar:
            pbar.update(1)
        return result

    async def __call__(
        self,
        messages_list: list[list[dict[str, str]]],
        request_kwargs: dict,
        max_concurrency: int = 0,
    ):
        sem = Semaphore(max_concurrency) if max_concurrency > 0 else None
        pbar = tqdm(total=len(messages_list))
        results = await gather(
            *[
                self._request_gpt(messages, request_kwargs, sem=sem, pbar=pbar)
                for messages in messages_list
            ]
        )
        pbar.close()
        return results
