"""
ragas_eval.py — RAGAS baseline evaluation for the MediCare Hospital Agent.

Run: python ragas_eval.py

Requires: pip install ragas datasets
"""

import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Ground truth QA pairs derived directly from the knowledge base
EVAL_DATASET = [
    {
        "question": "What are the OPD morning session timings?",
        "ground_truth": "Morning OPD sessions run from 8:00 AM to 1:00 PM, Monday to Saturday.",
    },
    {
        "question": "What is the consultation fee for a specialist?",
        "ground_truth": "Specialist consultations such as Cardiology, Neurology, and Gastroenterology cost Rs. 600 per consultation.",
    },
    {
        "question": "How do I book an appointment online?",
        "ground_truth": "Appointments can be booked online at www.medicarehyd.in, which is available 24/7.",
    },
    {
        "question": "What is the emergency helpline number?",
        "ground_truth": "The emergency helpline number is 040-99999999, operational 24 hours a day, 7 days a week.",
    },
    {
        "question": "What does the Basic Health Package include?",
        "ground_truth": (
            "The Basic Health Package costs Rs. 999 and includes CBC, blood sugar fasting, lipid profile, "
            "urine routine, BMI assessment, and physician consultation."
        ),
    },
]


def run_ragas_eval():
    from medicare_assistant.graph import build_graph, ask
    from datasets import Dataset

    print("Building agent...")
    app, _, _ = build_graph()

    rows = []
    print("\nRunning agent on evaluation questions...")
    for item in EVAL_DATASET:
        result = ask(app, item["question"], thread_id=f"ragas_{item['question'][:20]}")
        rows.append(
            {
                "question": item["question"],
                "answer": result.get("answer", ""),
                "contexts": [result.get("retrieved", "")],
                "ground_truth": item["ground_truth"],
            }
        )
        print(f"  Q: {item['question'][:60]}...")
        print(f"  A: {result.get('answer', '')[:80]}...\n")

    dataset = Dataset.from_list(rows)

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        print("Running RAGAS evaluation...")
        results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        print("\n=== RAGAS Baseline Scores ===")
        print(results)
        return results

    except ImportError:
        print("RAGAS not installed. Running manual faithfulness scoring...")
        _manual_faithfulness(rows, app)


def _manual_faithfulness(rows, app):
    """Fallback: LLM-based faithfulness scoring without RAGAS."""
    from medicare_assistant.graph import build_graph
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=0,
    )

    scores = []
    for row in rows:
        prompt = f"""Rate the faithfulness of this answer on a scale 0.0 to 1.0.
Context: {row['contexts'][0][:500]}
Answer: {row['answer']}
Reply with a single number only."""
        resp = llm.invoke(prompt)
        try:
            score = float(resp.content.strip())
        except ValueError:
            score = 0.5
        scores.append(score)
        print(f"  Q: {row['question'][:50]} | faithfulness={score:.2f}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nManual Average Faithfulness: {avg:.2f}")


if __name__ == "__main__":
    run_ragas_eval()
