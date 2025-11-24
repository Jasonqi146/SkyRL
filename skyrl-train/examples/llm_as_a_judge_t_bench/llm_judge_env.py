from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any
from typing import Dict
from omegaconf import DictConfig
from openai import OpenAI
import os
import re

PROMPT = """
You are an expert evaluation assistant for terminal/command-line tasks.

Your job is to evaluate whether a proposed step/action is **helpful and moves towards a successful outcome**.

You will be given:
1. **REFERENCE SOLUTION**: A working solution (not the only possible solution, just one valid approach)
2. **RUBRICS**: Granular quality criteria for the final outcome (at least 70 percent should be satisfied for success)
3. **PROPOSED STEP**: The action being evaluated

Your task:
- Evaluate if the proposed step is **constructive and helpful** towards achieving an outcome that satisfies at least 70 percent of the rubrics
- Consider whether the step aligns with good practices, even if it differs from the reference solution
- A step can be good even if it's not identical to the reference solution, as long as it makes meaningful progress
- Focus on: Is this step moving in the right direction? Is it a reasonable action to take given the task?

Scoring Guidelines:
- Score **1.0** if the step is clearly helpful and moves towards satisfying the rubrics
- Score **0.5** if the step is somewhat helpful but suboptimal or only partially correct
- Score **0.0** if the step is unhelpful, incorrect, or moves away from the goal

Instructions:
- You may provide internal reasoning or explanation before giving your final judgment
- Your final judgment must appear as a separate line at the end of your response, in the format:

### Final Score: <score>

where <score> is 0.0, 0.5, or 1.0

Do not include any explanation after the final score.
"""


class TBenchLLMJudgeEnv(BaseTextEnv):
    """
    T-Bench environment with LLM as judge.

    Use LLM to evaluate if each step is helpful towards achieving a good outcome
    based on reference solution and rubrics.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.reference_solution = extras["reward_spec"]["ground_truth"]
        
        # Get rubrics from extra_info if available
        self.rubrics = extras.get("extra_info", {}).get("rubrics", [])
        
        # Set up OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("`OPENAI_API_KEY` must be set for Llm as a judge env")
        self.llm_judge_client = OpenAI(base_url=env_config.base_url, api_key=openai_api_key)
        self.model = env_config.model

    def _format_rubrics(self) -> str:
        """Format rubrics into a readable string."""
        if not self.rubrics:
            return "No specific rubrics provided."
        
        formatted = []
        for i, rubric in enumerate(self.rubrics, 1):
            if isinstance(rubric, dict):
                # If rubric is a dict, extract relevant fields
                criteria = rubric.get('criteria', rubric.get('description', str(rubric)))
                points = rubric.get('points', '')
                formatted.append(f"{i}. {criteria}" + (f" (Points: {points})" if points else ""))
            else:
                formatted.append(f"{i}. {rubric}")
        
        return "\n".join(formatted)

    def _get_reward(self, action: str) -> float:
        rubrics_text = self._format_rubrics()
        
        message = (
            PROMPT + 
            f"\n\n**REFERENCE SOLUTION:**\n{self.reference_solution}\n\n"
            f"**RUBRICS (at least 70% should be satisfied):**\n{rubrics_text}\n\n"
            f"**PROPOSED STEP:**\n{action}\n\n"
            "**Your Evaluation:**"
        )

        try:
            response = self.llm_judge_client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": message}]
            )
            reply = response.choices[0].message.content.strip()

            # Try to parse score from "### Final Score: x"
            match = re.search(r"### Final Score:\s*([0-9]*\.?[0-9]+)", reply)
            if match:
                score = float(match.group(1))
                # Clamp score between 0 and 1
                return max(0.0, min(1.0, score))

            # Fallback: look for standalone numbers
            if reply.strip() in {"1", "1.0", "0.5", "0", "0.0"}:
                return float(reply.strip())

            print(f"Unrecognized reward output: {reply}")
            return 0.0

        except Exception as e:
            print(f"LLM Judge error: {type(e).__name__}: {e}")
            return 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward = self._get_reward(action)

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
