from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from llama_index.core.evaluation import EvaluationResult




def get_answers_source_nodes(responses)->Tuple[list, list]:

    answers = []
    sources = []
        
    for response in responses:
        answers.append(response.response)
        text_md = ""
        
        for n in response.source_nodes:
            
            text_md += (
                f"**Node ID:** {n.node.node_id}{chr(10)}"
                f"**Similarity:** {n.score}{chr(10)}"
                f"**Text:** {n.node.get_content()}{chr(10)}"
                f"**Metadata:** {n.node.metadata}{chr(10)}"
                f"~~~~{chr(10)}"
            )
        sources.append(text_md)
    
    return answers, sources

def get_summary_scores_df(
    eval_results_list: List[EvaluationResult], names: List[str], metric_keys: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get results df.

    Args:
        eval_results_list (List[EvaluationResult]):
            List of evaluation results.
        names (List[str]):
            Names of the evaluation results.
        metric_keys (List[str]):
            List of metric keys to get.

    """
    mean_dict = defaultdict(list)
    mean_dict["score"] = "Mean Score"
    
    sum_dict = defaultdict(list)
    sum_dict["score"] = "Total Score"

    for metric_key in metric_keys:
        for eval_results in eval_results_list:
            mean_score = np.array([r.score for r in eval_results[metric_key]]).mean()
            sum_score = np.array([r.score for r in eval_results[metric_key]]).sum()
            mean_dict[metric_key].append(mean_score)
            sum_dict[metric_key].append(sum_score)
    
    return pd.DataFrame(mean_dict), pd.DataFrame(sum_dict)


def get_eval_results_df(
    query_nums: List[str], expected_answers: List[str], results_arr: List[EvaluationResult], metric: Optional[str] = None
) -> pd.DataFrame:
    """Organizes EvaluationResults into a deep dataframe and computes the mean
    score.

    result:
        result_df: pd.DataFrame representing all the evaluation results
        
    """
    if len(query_nums) != len(results_arr):
        raise ValueError("names and results_arr must have same length.")

    qs = []
    ss = []
    fs = []
    rs = []
    cs = []
    for res in results_arr:
        qs.append(res.query)
        ss.append(res.score)
        fs.append(res.feedback)
        rs.append(res.response)
        cs.append(res.contexts)

    deep_df = pd.DataFrame(
        {
            "query_num": query_nums,
            "query": qs,
            "expected_answer": expected_answers,
            "generated_answer": rs,
            "score": ss,
            "feedback": fs,
            "contexts": cs
        }
    )
    
    return deep_df