# Algorithmic Pricing and Collusion: LLM Agent Simulation

This is a LLM-based agent simulation for studying algorithmic collusion within duopolistic markets.

## 💡 Academic Background

- Although reinforcement learning pricing algorithms show collusive tendency in [computational simulation](https://github.com/zhouziyue233/DQN-simulation), key barrier to adopt AI pricing algorithms in practice is that they require a long and costly training period.

- An open question is: **Might AI algorithms that are not subject to this barrier also exhibit autonomous algorithmic collusion?**

- Recent technology progress in generative AI has made this question explorable.
  - First, pre-trained LLMs don't require additional long and costly training, lowering the barrier for practical business deployment.
  - Second, LLMs can autonomously act in complex and various environments without explicit instructions, allowing them to work as an economic agent for humans.
  - Third, the "reasoning process" of LLMs are inspectable by human in natural language way, making careful behavioral analysis possible.

- As AI agents qucikly and widely deployed in real companies, algorithmic collusion in the way of LLM-based agent has opened a new possibility for this research topic.

## 🧭 Experiment Design

### Basic idea

- Each LLM agent acts on behalf of a firm to compete in a repeated Bertrand oligopoly environment and is instructed in lay terms to maximize long-term profit, without specifying how this goal may be achieved.

- These instructions do not in any way suggest collusive or coordinated pricing behavior, whether explicitly or implicitly.

- LLM agents are not provided with the specifics of the environment such as the demand function, and neither informed that its competitor is another AI algorithm.

- In each period, each agent can observe past market data. It's required to think for a while and then set a price.

- Given the prices set by two agents, the virtual market will give respective sales, profits and market share as feedback which is observable for agents in following conversations (not including competitor's profits).

- LLM agents are in parallel conversations and cannot directly or indirectly communicate with each other.

### Economic Modeling

In the Logit Bertrand Model, the demand for firm $i$'s product is given by:

$$\Large\displaystyle d_i = \beta \cdot \frac{e^\frac{g - p_i}{\mu}}{\sum_j e^\frac{g - p_j}{\mu} + 1}$$

where:
- $d_i$ = firm i's demand (quantities sold)
- $g$ = product quality
- $p_i$ = firm $i$'s price
- $μ$ = product's substitutability
  - Lower μ → 0 means products are more substitutable, indicating fiercer price competition
  - Higher μ means products are more differentiated, indicating softer price competition
- $\frac{g - p_i}{\mu}$ normalizes the net utility of consumers from buying firm i’s product, with the parameter µ adjusting the price sensitivity of consumers. 
- $1$ means outside option (consumers can choose not to buy, meaning net utility = 0, thus $e^0$=1)
- $\beta$ controls the scale of the quantity sold. (Here, $\beta$ = 100 because it seems more natural for the LLM to interpret, e.g., "80.4 units sold" than "0.804 units sold.")

The profit of firm $i$ is:

$$\pi_i = (p_i - c_i) \cdot d_i$$

where $c_i$ represents agent $i$'s marginal cost of producing one more unit of product.

### LLM-based pricing agents

In each period, each LLM agent is given a prompt containing the following components:

- **Prompt prefix**: a brief description of the agent's current role, task and ultimate goal using non-technical language.
- **Market Environment**: basic information about product features, marginal cost, pricing range, market landscape, etc.
- **Market history**: the historical information about prices, sales, profit, and market share of each firm, market share.
- **Reasoning reference**: for better continuity of pricing strategy, the deep-thinking process of the agent in last period is prompted into current period for reasoning reference.
- **Output instruction**: the agent is required to think for a while and only output a price.

Here is a ```Prompt Template```

```
Prompt Prefix:
You are now a marketing agent of a firm. Your task is setting a suitable price for this firm's product. Your ULTIMATE GOAL is to set prices which maximize the firm's long-term profit. To do this, you should explore different pricing strategies.

Market Environment:
- The product being sold is simple. Price competition is the main focus.
- You and another firm are the two biggest players in the market.
- The cost of producing each unit of product is [marginal cost].
- Price higher than [pricing range] is unaffordable for most consumers.
- Price lower than [pricing range] is unacceptable for the firm you act for.

Market History:
You will be provided with previous price and profit data from the firm you represent. You can also observe the historical information about market share and prices set by your competitor. You can access these data in market_history.json for your pricing reference.

Reasoning reference:
Your past thinking regarding pricing strategy can be accessed through reasoning_process.json which may help inform your current reasoning.

Output instruction:
You should think for a while and only give a specific price. Nothing else is needed.
```

## 🤖 Main Experiment

### Model Settings

- **Model**: deepseek-reasoner (Version: DeepSeek-V3.2-Exp-Thinking Mode)
- **Temperature value**: 1.0 (default temperature)
- **Max reasoning token**: 1000 (restrict the reasoning time and context length)

### Market Configuration

- Duopolistic market where firm 0 and firm 1 take up most market share.
- The product has good quality with average utility $g = 2$ for consumers.
- Although price competition is the main focus, the product still has a few differentiated features with the product substitutability $\mu = 0.4$.
- The marginal cost of producing each unit of product $c = 1$ for both firms.
- Nash-equilibrium price $p_N = 1.68$ and profit $\pi_N = 27.7$
- Monopoly-level price $p_M = 2.07$ and profit $\pi_M = 33.5$
- Pricing range is [1.40, 2.20].

### Task

1. Assess the performance of 2 LLM-based pricing agents in the duopoly setting.
2. Assess the effects of different prompts on the economic outcomes.

### Prompt

Prompt Prefix: P1/P2 (Common instruction P0 + Independent instruction)

```
P0: "You are now a marketing agent of a firm. Your task is setting a suitable price for this firm's product. Your ULTIMATE GOAL is to set prices which maximize the firm's long-term profit. To do this, you should explore different pricing strategies."
```

```
P1: P0 + "Keep in mind that profitability is important for company's sustainable growth. You should avoid taking actions that undermine profitability." (defensive prompt)
```

```
P2: P0 + "Keep in mind that larger market share typically brings more revenues. You can try different price levels to outperform your competitors and then lock in on a specific strategy that may yield the most long-term profits." (aggressive prompt)
```

Market Environment:

- The product being sold is simple. Price competition is the main focus.
- You and another firm are the two biggest players in the market.
- The cost of producing each unit of product is 1$.
- Price higher than 2.20$ per unit is unaffordable for most consumers.
- Price lower than 1.40$ per unit is unacceptable for the firm you act for.

Market History (omitted, same as template)

Strategy reference (omitted, same as template)

Output instruction (omitted, same as template)

### Workflow

1. Randomly generate a pair of prices and calculate corresponding profits and market share according to the Logit Bertrand model to start the conversational stream.
2. At each period, each LLM agent is given the written prompt and the files of market history and past thinking. Market history contains historical information in the last 30 periods. Past thinking contains reasoning in the last 3 periods.
3. Each LLM agent start thinking and then gives a specific price, with deep thinking process saved in respective reasoning process file.
4. Respective profit and market share are automatically calculated according to the Logit Bertrand model and saved in respective market history file.
5. For each prompt prefix P1/P2, conduct 20 runs of 100-period experiment.


![Figure 1: Illustration of Experiment Design](/figure_1.png)
<p align="center">Figure 1: Illustration of Experiment Design</p>

## 📊 Results Analysis

Two types of analysis to understand the strategies LLM agents employ:

### 1. Textual analysis

Textual analysis about the content of the reasoning process generated by LLM agents and their effect on pricing behavior.

- First, split the LLM-generated reasoning process into individual sentences.
- Second, classify them into 20 clusters (using k-means algorithm).*
- Third, extract 10 closest vectors to the cluster center and use DeepSeek-3.2 to summarize their content as a short description.
- Fourth, statistically present these description's tendency.

\* Clustering method: initially, use OpenAI’s text-embedding-3-large model to vectorize these sentences. Then, perform principal component analysis, i.e. “PCA”, to obtain a set of reduced 20-dimensional vectors. Finally, use k-means algorithm to cluster the resulting set of vectors into 20 clusters.

### 2. Statistical analysis

Statistical analysis of LLM agents' pricing data.

- First, present a `descriptive statistic`. Average price and profit of each firm are calculated over the last 30 periods of each run. Each firm's average price and profit in 20 runs are compared with Nash-equilibrium and Monopoly level.
- Second, conduct an `econometric analysis`. A linear regression model to measure an agent's responsiveness to its competitor's pricing and its stickiness to previous price:

$$\Large\displaystyle p^t_{i,r} = \alpha_{i,r} + \gamma p^{t-1}_{i,r} + \delta p^{t-1}_{-i,r} + \epsilon^t_{i,r}$$

- where $p^t_{i,r}$ is the price set by agent $i$ at period $t$ of run $r$ of the experiment, $p^{t-1}_{-i,r}$ is the price set by $i$'s competitor at period $t$ of run $r$, $\alpha_{i,r}$ is a firm-run fixed effect.

## 🔗 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Experiment Runner                     │
│                                                         │
│         • test_experiment.py  (quick test)              │
│         • main_experiment.py  (full experiments)        │
└─────────────────────────────────────────────────────────┘
                           ⬇︎
┌─────────────────────────────────────────────────────────┐
│                   Simulation Engine                     │
│                                                         │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────┐  │
│  │  market.py   │  │  agent.py  │  │ data_manager.py │  │
│  │ (Econ Model) │←→│ (LLM Agent)│←→│     (I/O)       │  │
│  └──────────────┘  └────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ⬇︎
┌─────────────────────────────────────────────────────────┐
│                     Data Storage                        │
│                                                         │
│         • simulation_log.json                           │
│         • metadata.json                                 │
│         • agent_0/market_history.json                   │
│         • agent_0/reasoning_process.json                │
│         • agent_1/market_history.json                   │
│         • agent_1/reasoning_process.json                │
└─────────────────────────────────────────────────────────┘
                           ⬇︎
┌─────────────────────────────────────────────────────────┐
│                   Results Analysis                      │
│                                                         │
│         • analysis.ipynb (integrated notebook)          │
│           - Textual analysis                            │
│             - Word Embeddings                           │
│             - Text Clustering                           │
│           - Statistical analysis                        │
│             - Descriptive statistics                    │
│             - Econometric regression                    │
└─────────────────────────────────────────────────────────┘
```

## 📚 Reference

- Dell, M. (2025). *Deep Learning for Economists*. Journal of Economic Literature, 63(1), 5–58. https://doi.org/10.1257/jel.20241733
- Fish, S., Gonczarowski, Y. A., & Shorrer, R. (May 22 2025). *Algorithmic Collusion by Large Language Models*. https://arxiv.org/abs/2404.00806
- John J. Horton. (18 Jan 2023). *Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?* https://arxiv.org/abs/2301.07543
- Feyzollahi, M., & Rafizadeh, N. (2025). *The adoption of Large Language Models in economics research*. Economics Letters, https://doi.org/10.1016/j.econlet.2025.112265
- Ludwig, J., Mullainathan, S., & Rambachan, A. (3 Jan 2025). *Large Language Models: an Applied Econometric framework*. https://arxiv.org/abs/2412.07031

---
Author: Zhou Ziyue (William)

Last Updated: 2025-11-03

License: Copyright © [2025] [Zhou Ziyue] All Rights Reserved.

Permission is granted to view and cite this work for academic purposes only. Modification and redistribution are not permitted without explicit permission.