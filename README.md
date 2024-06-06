# Reinforcement Learning From Human Feedback

Reinforcement learning (RL) refers to a sub-field of machine learning that enables AI-based systems to take actions in a dynamic environment through trial and error to maximize the collective rewards based on the feedback generated for individual activities.

Proximal policy optimization (PPO) is an algorithm in the field of reinforcement learning that trains a computer agent's decision function to accomplish difficult tasks.

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR1VhfIprSiO7zBkm0a3yyaju8E2194bbl9Ww&usqp=CAU)

##**What is reinforcement learning from human feedback?**
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJLH87EqSSwnXfd4724ftjSAUNv7wiEftpqg&usqp=CAU)

Reinforcement learning from human feedback (RLHF) is a machine learning (ML) technique that uses human feedback to optimize ML models to self-learn more efficiently. Reinforcement learning (RL) techniques train software to make decisions that maximize rewards, making their outcomes more accurate.

Reinforcement means you are increasing a behavior, and punishment means you are decreasing a behavior. Reinforcement can be positive or negative, and punishment can also be positive or negative. All reinforcers (positive or negative) increase the likelihood of a behavioral response.

ChatGPT is a Large Language Model (LLM) optimized for dialogue. It is built on top of GPT 3.5 using Reinforcement Learning from Human Feedback (RLHF).

###**How does ChatGPT use human feedback?**
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuSOLx0NCs2kF7E2gu5kTCSgk_DzfsYGALtw&usqp=CAU)
Periodically, the agent presents two video clips of its behavior to the human evaluator, who then decides which clip is closest to fulfilling the goal of a backflip. The agent then uses this feedback to gradually build a model of the goal and the reward function that best explains the human's judgments.

###**How does RLHF work?**
Training an LLM with RLHF typically occurs in four phases:

**Pre-training models**
RLHF is generally employed to fine-tune and optimize a pre-trained model, rather than as an end-to-end training method. For example, InstructGPT used RLHF to enhance the pre-existing GPT—that is, Generative Pre-trained Transformer—model. In its release announcement for InstructGPT, OpenAI stated that “one way of thinking about this process is that it ‘unlocks’ capabilities that GPT-3 already had, but were difficult to elicit through prompt engineering alone.”5  

Pre-training remains by far the most resource-intensive phase of RLHF. OpenAI noted that the RLHF training process for InstructGPT entailed less than 2 percent of the computation and data needed for the pre-training of GPT-3.

**Supervised fine-tuning**
Prior to the start of explicit reinforcement learning, supervised fine-tuning (SFT) is used to prime the model to generate its responses in the format expected by users.

As alluded to earlier, the LLM pre-training process optimizes models for completion: predicting the next words in a sequence began with the user’s prompt by replicating linguistic patterns learned during model pre-training. Sometimes, LLMs won’t complete a sequence in the way a user wants: for example, if a user’s prompt is, “teach me how to make a resumé,” the LLM might respond with “using Microsoft Word.” It’s a valid way to complete the sentence, but not aligned with user’s goal.

SFT therefore uses supervised learning to train models to respond appropriately to different kinds of prompts. Human experts create labeled examples, following the format (prompt, response), to demonstrate how to respond to prompts for different use cases, like question answering, summarization or translation.

This demonstration data, while powerful, is time-consuming and expensive to generate. Rather than create bespoke new examples, DeepMind introduced the approach of "applying a filtering heuristic based on a common written dialogue format (‘interview transcript’ style)” to isolate suitable prompt/response example pairings from within their MassiveWeb dataset.9

**Reward model training**
For human feedback to power a reward function in reinforcement learning, a reward model is needed to translate human preference into a numerical reward signal. Designing an effective reward model is a crucial step in RLHF, as no straightforward mathematical or logical formula exists to feasibly define subjective human values.

The main purpose of this phase is to provide the reward model with sufficient training data, comprised of direct feedback from human evaluators, to help the model learn to mimic the way human preferences allocate rewards to different kinds of model responses. This allows for training to continue offline without a human in the loop.

A reward model must intake a sequence of text and output a scalar reward value that predicts, numerically, how much a human user would reward (or penalize) that text. This output being a scalar value is essential for the output of the reward model to be integrated with other components of the RL algorithm.

While it might seem most intuitive to simply have human evaluators express their opinion of each model response in scalar form—like rating the response on a scale of one (worst) to ten (best)—it’s prohibitively difficult to get all human raters aligned on the relative value of a given score, much less get human raters aligned on what constitutes a “good” or “bad” response in a vacuum. This can make direct scalar rating noisy and challenging to calibrate.

Instead, a rating system is usually constructed by comparing human feedback for different model outputs. A common method is to have users compare two analogous text sequences—like the output of two different language models responding to the same prompt—in head-to-head matchups, then use an Elo rating system to generate an aggregated ranking of each bit of generated text relative to one another. A simple system might allow users to “thumbs up” or “thumbs down” each output, with outputs then being ranked by their relative favorability. More complex systems might ask labelers to provide an overall rating and answer categorical questions about the flaws of each response, then algorithmically aggregate this feedback into a weighted quality score.

The outcomes of whichever ranking systems are ultimately normalized into a scalar reward signal to inform reward model training.

**Policy optimization**
The final hurdle of RLHF is determining how—and how much—the reward model should be used to update the AI agent’s policy. One of the most successful algorithms used for the reward function that updates RL models is proximal policy optimization (PPO).

Unlike most machine learning and neural network model architectures, which use gradient descent to minimize their loss function and yield the smallest possible error, reinforcement learning algorithms often use gradient ascent to maximize reward.

However, if the reward function is used to train the LLM without any guardrails, the language model may dramatically change its weights to the point of outputting gibberish in an effort to “game” the reward model. PPO provides a more stable means of updating the AI agent’s policy by limiting how much the policy can be updated in each training iteration.

First, a copy of the initial model is created and its trainable weights are frozen. The PPO algorithm calculates a range of [1-ε, 1+ε], in which ε is a hyperparameter that roughly determines how far the new (updated) policy is allowed to stray from the old (frozen) policy. Then, it calculates a probability ratio: the ratio of the probability of a given action being taken by the old policy vs. the probability of that action being taken by the new policy. If the probability ratio is greater than 1+ε (or below 1-ε), the magnitude of the policy update may be clipped to prevent any steep changes that may destabilize the entire model.

The introduction of PPO provided an attractive alternative to its predecessor, trust region policy optimization (TRPO), which provides similar benefits but is more complicated and computationally expensive than PPO. While other policy optimization frameworks like advantage actor-critic (A2C) are also viable, PPO is often favored as a simple and cost-effective methodology.


###**Limitations of RLHF**
Though RLHF models have demonstrated impressive results in training AI agents for complex tasks from robotics and video games to NLP, using RLHF is not without its limitations.

- Human preference data is expensive. The need to gather firsthand human input can create a costly bottleneck that limits the scalability of the RLHF process. Both Anthropic10 and Google11 have proposed methods of reinforcement learning from AI feedback (RLAIF), replacing some or all human feedback by having another LLM evaluate model responses, that have yielded results comparable to those of RLHF.

- Human input is highly subjective. It’s difficult, if not impossible, to establish firm consensus on what constitutes “high-quality” output, as human annotators will often disagree on not only alleged facts, but also what “appropriate” model behavior should mean. Human disagreement thus precludes the realization of a genuine “ground truth” against which model performance can judged.

- Human evaluators can be fallible, or even intentionally adversarial and malicious. Whether reflecting genuine contrarian views or intentionally trolling the learning process, human guidance to the model is not always provided in good faith. In a 2016 paper, Wolf, et al posited that toxic behavior should be a fundamental expectation of human-bot interactions and suggested the need for a method to assess the credibility of human input.12 In 2022, Meta AI released a paper on adversarial human input (link resides outside ibm.com) studying automated methods “to gain maximum learning efficiency from high quality data, while simultaneously being maximally robust to low quality and adversarial data.” The paper identifies various
“troll” archetypes and the different ways they distort feedback data.

- RLHF risks overfitting and bias. If human feedback is gathered from an overly narrow demographic, the model may demonstrate performance issues when used by different groups or prompted on subject matters for which the human evaluators hold certain biases.
