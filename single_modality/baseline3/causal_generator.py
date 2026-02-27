"""
SemEval 2026 Task 12: Abductive Event Reasoning
Causal Knowledge Generator using Qwen3-0.5B

This module provides a two-stage inference pipeline:
1. Local Qwen3-0.5B generates causal knowledge (cause hypotheses, temporal context)
2. DeepSeek API performs final reasoning with enriched context

Usage:
    from causal_generator import CausalKnowledgeGenerator, TwoStageReasoner

    # Initialize generator (local model)
    generator = CausalKnowledgeGenerator("./output/causal_grpo")

    # Initialize two-stage reasoner
    reasoner = TwoStageReasoner(
        generator=generator,
        deepseek_api_key="your-api-key"
    )

    # Inference
    result = reasoner.predict(target_event, options)

Requirements:
    pip install transformers torch accelerate
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class CausalKnowledge:
    """Generated causal knowledge structure"""
    cause_hypotheses: List[str]         # Potential causes
    temporal_context: str               # Temporal analysis
    causal_chains: List[str]            # Causal reasoning chains
    confidence_scores: List[float]      # Confidence for each hypothesis
    raw_generation: str                 # Raw model output


class CausalKnowledgeGenerator:
    """
    Qwen3-0.5B based Causal Knowledge Generator

    Generates causal hypotheses and temporal context for target events.
    Can be fine-tuned with SFT and GRPO for domain-specific causal reasoning.
    """

    SYSTEM_PROMPT = """You are an expert in causal reasoning and event analysis.
Given a target event, analyze the potential causes based on:
1. Temporal precedence (causes must precede effects)
2. Direct causal relationships (not just correlation)
3. Contextual relevance (matching domain and scope)

Provide clear causal hypotheses with reasoning."""

    GENERATION_TEMPLATE = """Target Event: {target_event}

Analyze this event and identify:
1. Most likely direct causes
2. Temporal context
3. Causal reasoning chain

{options_context}

Provide your causal analysis:"""

    MCQA_TEMPLATE = """Target Event: {target_event}

Options:
{options_text}

Based on causal reasoning principles:
- Temporal precedence: Which events occurred before the target?
- Direct causation: Which events directly led to the target?
- Relevance: Which events are most contextually related?

Analyze each option and identify the direct cause(s):"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 512,
        use_lora: bool = True,
        base_model: str = "Qwen/Qwen3-0.5B-Instruct"
    ):
        """
        Initialize the causal knowledge generator

        Args:
            model_path: Path to fine-tuned model or LoRA adapter
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype for model
            max_length: Maximum generation length
            use_lora: Whether model_path contains LoRA adapter
            base_model: Base model name (used if use_lora=True)
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length

        print(f"Loading causal knowledge generator from {model_path}...")

        # Load tokenizer
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if use_lora and Path(model_path).exists():
            # Load base model + LoRA adapter
            print(f"Loading base model: {base_model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            print(f"Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model = self.model.merge_and_unload()
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        self.model.eval()
        print("Causal knowledge generator loaded successfully!")

    def _build_messages(
        self,
        target_event: str,
        options: Dict[str, str] = None,
        mode: str = "generation"
    ) -> List[Dict[str, str]]:
        """Build chat messages for generation"""
        if mode == "mcqa" and options:
            options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
            user_content = self.MCQA_TEMPLATE.format(
                target_event=target_event,
                options_text=options_text
            )
        else:
            options_context = ""
            if options:
                options_context = "Consider these potential causes:\n"
                for k, v in options.items():
                    options_context += f"- {k}: {v}\n"

            user_content = self.GENERATION_TEMPLATE.format(
                target_event=target_event,
                options_context=options_context
            )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

    @torch.no_grad()
    def generate(
        self,
        target_event: str,
        options: Dict[str, str] = None,
        mode: str = "generation",
        num_hypotheses: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> CausalKnowledge:
        """
        Generate causal knowledge for a target event

        Args:
            target_event: The effect event to analyze
            options: Optional dict of options {"A": "...", "B": "..."}
            mode: "generation" (free-form) or "mcqa" (option-based)
            num_hypotheses: Number of cause hypotheses to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding

        Returns:
            CausalKnowledge object with generated hypotheses
        """
        messages = self._build_messages(target_event, options, mode)

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse output
        return self._parse_generation(raw_output, options)

    def _parse_generation(
        self,
        raw_output: str,
        options: Dict[str, str] = None
    ) -> CausalKnowledge:
        """Parse raw generation into structured causal knowledge"""
        cause_hypotheses = []
        temporal_context = ""
        causal_chains = []
        confidence_scores = []

        lines = raw_output.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse cause hypotheses
            if line.startswith(('1.', '2.', '3.', '-', '*', 'Cause:', 'Direct cause:')):
                cause = line.lstrip('0123456789.-* ').strip()
                if cause:
                    cause_hypotheses.append(cause)
                    confidence_scores.append(0.5)  # Default confidence

            # Parse temporal context
            elif 'temporal' in line.lower() or 'before' in line.lower() or 'after' in line.lower():
                temporal_context += line + " "

            # Parse causal chains
            elif '→' in line or '->' in line or 'leads to' in line.lower():
                causal_chains.append(line)

        # If MCQA mode, extract option letters
        if options:
            for opt_letter in ['A', 'B', 'C', 'D']:
                if opt_letter in raw_output.upper()[:50]:
                    if opt_letter in options:
                        cause_hypotheses.insert(0, f"[{opt_letter}] {options[opt_letter]}")
                        confidence_scores.insert(0, 0.8)

        return CausalKnowledge(
            cause_hypotheses=cause_hypotheses[:5],
            temporal_context=temporal_context.strip(),
            causal_chains=causal_chains[:3],
            confidence_scores=confidence_scores[:5],
            raw_generation=raw_output
        )

    def generate_batch(
        self,
        questions: List[Dict[str, Any]],
        mode: str = "mcqa"
    ) -> List[CausalKnowledge]:
        """Generate causal knowledge for multiple questions"""
        results = []
        for q in questions:
            target_event = q["target_event"]
            options = {
                opt: q[f"option_{opt}"]
                for opt in ["A", "B", "C", "D"]
                if f"option_{opt}" in q
            }
            result = self.generate(target_event, options, mode)
            results.append(result)
        return results


class TwoStageReasoner:
    """
    Two-Stage Reasoning Pipeline:
    1. Local Qwen3-0.5B generates causal knowledge
    2. DeepSeek API performs final reasoning

    This allows:
    - Fast local preprocessing with domain-tuned model
    - High-quality final reasoning with larger model
    - Reduced API costs by enriching context locally
    """

    DEEPSEEK_PROMPT_TEMPLATE = """You are an expert in causal reasoning for news events.

Target Event: {target_event}

Options:
{options_text}

=== Causal Analysis (from specialized model) ===
{causal_knowledge}
===

Based on the causal analysis above and your reasoning:
1. Which option(s) represent the DIRECT cause of the target event?
2. Consider temporal precedence and causal relationships.

Answer with ONLY the letter(s) of the correct option(s), separated by commas if multiple.
For example: "A" or "A, C"

Your answer:"""

    def __init__(
        self,
        generator: CausalKnowledgeGenerator,
        deepseek_api_key: str = None,
        deepseek_base_url: str = "https://api.deepseek.com/v1",
        deepseek_model: str = "deepseek-chat"
    ):
        """
        Initialize two-stage reasoner

        Args:
            generator: CausalKnowledgeGenerator instance
            deepseek_api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env)
            deepseek_base_url: DeepSeek API base URL
            deepseek_model: DeepSeek model name
        """
        self.generator = generator
        self.api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = deepseek_base_url
        self.model = deepseek_model

        if not self.api_key:
            print("Warning: No DeepSeek API key provided. Set DEEPSEEK_API_KEY env var.")

    def _format_causal_knowledge(self, knowledge: CausalKnowledge) -> str:
        """Format causal knowledge for DeepSeek prompt"""
        parts = []

        if knowledge.cause_hypotheses:
            parts.append("Potential Causes:")
            for i, cause in enumerate(knowledge.cause_hypotheses, 1):
                conf = knowledge.confidence_scores[i-1] if i <= len(knowledge.confidence_scores) else 0.5
                parts.append(f"  {i}. {cause} (confidence: {conf:.1f})")

        if knowledge.temporal_context:
            parts.append(f"\nTemporal Context: {knowledge.temporal_context}")

        if knowledge.causal_chains:
            parts.append("\nCausal Chains:")
            for chain in knowledge.causal_chains:
                parts.append(f"  - {chain}")

        return "\n".join(parts) if parts else "No specific causal analysis available."

    def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek API"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a causal reasoning expert. Answer concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )

            return response.choices[0].message.content.strip()

        except ImportError:
            print("Warning: openai package not installed. Using mock response.")
            return "A"
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return ""

    def predict(
        self,
        target_event: str,
        options: Dict[str, str],
        use_deepseek: bool = True
    ) -> Dict[str, Any]:
        """
        Predict the direct cause(s) using two-stage reasoning

        Args:
            target_event: The effect event
            options: Dict of options {"A": "...", "B": "...", ...}
            use_deepseek: Whether to use DeepSeek for final reasoning

        Returns:
            Dict with prediction, causal_knowledge, and raw outputs
        """
        # Stage 1: Local causal knowledge generation
        knowledge = self.generator.generate(
            target_event=target_event,
            options=options,
            mode="mcqa"
        )

        result = {
            "target_event": target_event,
            "options": options,
            "causal_knowledge": {
                "cause_hypotheses": knowledge.cause_hypotheses,
                "temporal_context": knowledge.temporal_context,
                "causal_chains": knowledge.causal_chains,
                "confidence_scores": knowledge.confidence_scores,
            },
            "local_raw": knowledge.raw_generation
        }

        # Stage 2: DeepSeek reasoning
        if use_deepseek and self.api_key:
            options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
            causal_text = self._format_causal_knowledge(knowledge)

            prompt = self.DEEPSEEK_PROMPT_TEMPLATE.format(
                target_event=target_event,
                options_text=options_text,
                causal_knowledge=causal_text
            )

            deepseek_response = self._call_deepseek(prompt)
            result["prediction"] = self._parse_prediction(deepseek_response)
            result["deepseek_raw"] = deepseek_response
        else:
            # Fallback: use local model's prediction
            result["prediction"] = self._extract_local_prediction(knowledge, options)
            result["deepseek_raw"] = None

        return result

    def _parse_prediction(self, response: str) -> List[str]:
        """Parse prediction from DeepSeek response"""
        predictions = []
        response_upper = response.upper()

        for char in "ABCD":
            if char in response_upper:
                # Check it's a standalone letter, not part of a word
                idx = response_upper.find(char)
                if idx >= 0:
                    before_ok = idx == 0 or not response_upper[idx-1].isalpha()
                    after_ok = idx == len(response_upper)-1 or not response_upper[idx+1].isalpha()
                    if before_ok and after_ok:
                        predictions.append(char)

        return predictions if predictions else ["A"]  # Default to A if parsing fails

    def _extract_local_prediction(
        self,
        knowledge: CausalKnowledge,
        options: Dict[str, str]
    ) -> List[str]:
        """Extract prediction from local model's output"""
        predictions = []

        # Check raw generation for option letters
        raw_upper = knowledge.raw_generation.upper()
        for char in "ABCD":
            if f"[{char}]" in raw_upper or f"OPTION {char}" in raw_upper:
                predictions.append(char)

        # If no explicit options found, match hypotheses to options
        if not predictions and knowledge.cause_hypotheses:
            for opt_letter, opt_text in options.items():
                opt_lower = opt_text.lower()
                for hypothesis in knowledge.cause_hypotheses:
                    if opt_lower[:50] in hypothesis.lower() or hypothesis.lower()[:50] in opt_lower:
                        predictions.append(opt_letter)
                        break

        return predictions if predictions else ["A"]

    def predict_batch(
        self,
        questions: List[Dict[str, Any]],
        use_deepseek: bool = True
    ) -> List[Dict[str, Any]]:
        """Predict for multiple questions"""
        results = []
        for q in questions:
            target_event = q["target_event"]
            options = {
                opt: q[f"option_{opt}"]
                for opt in ["A", "B", "C", "D"]
                if f"option_{opt}" in q
            }
            result = self.predict(target_event, options, use_deepseek)
            result["question_id"] = q.get("id", "unknown")
            results.append(result)
        return results


def main():
    """Demo usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Causal Knowledge Generator Demo")
    parser.add_argument("--model", type=str, default="./output/causal_grpo",
                        help="Path to fine-tuned model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.5B-Instruct",
                        help="Base model name")
    parser.add_argument("--use-lora", action="store_true", default=True,
                        help="Model path contains LoRA adapter")
    parser.add_argument("--deepseek-key", type=str, default=None,
                        help="DeepSeek API key")
    args = parser.parse_args()

    # Example question
    target_event = "Iran launched ballistic missile attacks against US military bases in Iraq."
    options = {
        "A": "On December 29, US forces conducted airstrikes against Kataib Hezbollah positions.",
        "B": "After 2006, Muhandis founded Kataib Hezbollah with Iranian support.",
        "C": "On December 27, Kataib Hezbollah attacked a US base killing a contractor.",
        "D": "A U.S. drone strike killed Iranian General Qassem Soleimani on January 3."
    }

    print("=" * 60)
    print("Causal Knowledge Generator Demo")
    print("=" * 60)
    print(f"\nTarget Event: {target_event[:100]}...")
    print(f"\nOptions:")
    for k, v in options.items():
        print(f"  {k}. {v[:60]}...")

    # Check if model exists
    if not Path(args.model).exists():
        print(f"\nModel not found at {args.model}")
        print("Please train the model first using:")
        print("  1. python data_preparation/extract_causal_triples.py -i <questions.jsonl>")
        print("  2. python data_preparation/prepare_sft_data.py -i ./data/causal_triples.jsonl")
        print("  3. python training/run_sft.py --data ./data/causal_sft_data.jsonl")
        print("  4. python data_preparation/prepare_grpo_data.py -i ./data/causal_triples.jsonl")
        print("  5. python training/run_grpo.py --model ./output/causal_sft --data ./data/causal_grpo_data.jsonl")
        return

    # Initialize generator
    generator = CausalKnowledgeGenerator(
        model_path=args.model,
        base_model=args.base_model,
        use_lora=args.use_lora
    )

    # Generate causal knowledge
    print("\n" + "=" * 60)
    print("Stage 1: Local Causal Knowledge Generation")
    print("=" * 60)

    knowledge = generator.generate(target_event, options, mode="mcqa")

    print(f"\nCause Hypotheses:")
    for i, (cause, conf) in enumerate(zip(knowledge.cause_hypotheses, knowledge.confidence_scores), 1):
        print(f"  {i}. {cause[:80]}... (conf: {conf:.2f})")

    if knowledge.temporal_context:
        print(f"\nTemporal Context: {knowledge.temporal_context[:200]}...")

    if knowledge.causal_chains:
        print(f"\nCausal Chains:")
        for chain in knowledge.causal_chains:
            print(f"  - {chain[:80]}...")

    # Two-stage reasoning
    if args.deepseek_key:
        print("\n" + "=" * 60)
        print("Stage 2: DeepSeek Final Reasoning")
        print("=" * 60)

        reasoner = TwoStageReasoner(
            generator=generator,
            deepseek_api_key=args.deepseek_key
        )

        result = reasoner.predict(target_event, options)
        print(f"\nFinal Prediction: {result['prediction']}")
        print(f"DeepSeek Response: {result['deepseek_raw']}")
    else:
        print("\n(Skipping DeepSeek stage - no API key provided)")
        print("Set --deepseek-key or DEEPSEEK_API_KEY environment variable")


if __name__ == "__main__":
    main()
