Please scaffold this repo with the following information. We will be conducting config exploration experiments for a paper on blis @../inference-sim/. We will compare against several different tools:

1. LLMServingSim: https://github.com/casys-kaist/LLMServingSim. Commit f4ab208cf1db1c41a81401ed0e58752dfc960fb1
2. AIConfigurator: a pypi package, aiconfigurator=0.4.0, repo https://github.com/ai-dynamo/aiconfigurator/tree/v0.4.0
3. Vidur: https://github.com/microsoft/vidur commit 8383d29
4. llm-optimizer: https://github.com/bentoml/llm-optimizer commit bb82d22
5. inference-sim: https://github.com/inference-sim/inference-sim commit b05154c

I'd like you to clone them as submodules in this repo with the correct tags. For aiconfigurator, since it's a pypi package, we can also have it pinned in requirements.txt. The reason why I want them as submodules is so that we can run agents to understand the logic of the repo better if we have specific questions. Would submodules work like that? Is this clear?

